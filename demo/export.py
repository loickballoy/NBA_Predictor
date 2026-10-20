"""
Démo publique statique : `python -m demo.export --out docs`

Exporte le modèle en production (projections + explications) dans un fichier data.json et la
page web en mode statique, à publier sur GitHub Pages. Aucun appel à l'API au moment de la
visite : zéro coût, zéro surface d'attaque, et la démo reste en ligne même si GCP est éteint.

Les explications manquantes sont générées (et mises en cache dans le bucket) si ANTHROPIC_API_KEY
est défini ; sinon le repli sans IA est utilisé.
"""
import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from api.explain import Explainer, build_facts
from api.store import ModelStore
from nba_api.stats.static import players as nba_players

STATIC_PAGE = Path(__file__).resolve().parents[1] / "api" / "static" / "index.html"


def clean(v):
    """NaN n'existe pas en JSON : null à la place (comme api.main.clean)."""
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def log(severity, message, **fields):
    if fields.get("event") == "llm_call" and not fields.get("cached"):
        print(f"  {fields.get('player_id')}: {'ok' if fields.get('valid') else fields.get('error')} "
              f"({fields.get('cost_eur', 0):.5f} €)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs")
    parser.add_argument("--limit", type=int, default=0,
                        help="garder les N joueurs avec le plus de minutes la saison passée (0 = tous)")
    args = parser.parse_args()

    store = ModelStore(os.environ["STORAGE_URI"])
    snap = store.get()
    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
    explainer = Explainer(client, store.storage, log, model=os.environ.get("LLM_MODEL", "claude-haiku-5-5"))

    preds = snap.predictions
    if args.limit:
        preds = preds.sort_values("MIN_PER_GAME_last", ascending=False).head(args.limit)

    players, total_cost = [], 0.0
    stats = [c[: -len("_pred")] for c in preds.columns if c.endswith("_pred")]
    for pid, row in preds.iterrows():
        pid = int(pid)
        info = nba_players.find_player_by_id(pid)
        name = info["full_name"] if info else f"Joueur {pid}"
        facts = build_facts(row, name, pid)
        r = explainer.explain(snap.version, pid, facts)
        total_cost += r.cost_eur
        players.append({
            "player_id": pid, "player_name": name, "season": facts["saison_projetee"],
            "model_version": snap.version,
            "stats": {s: {"predicted": round(float(row[f"{s}_pred"]), 2),
                          "last_season": last if (last := row.get(f"{s}_last")) == last and last is not None
                          else None}
                      for s in stats},
            "explanation": {"source": r.source if r.source != "cache" else "claude",
                            **r.explanation.model_dump()},
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    test = snap.metrics["test"]
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"version": snap.version, "promoted_at": snap.production["promoted_at"],
                  "validation_score": snap.production["validation_score"],
                  "previous_version": snap.production.get("previous_version"),
                  "architecture": snap.metrics["model"]["architecture"],
                  # mêmes champs que GET /model : la page lit l'un ou l'autre sans distinction
                  "test_metrics": {s: {"mae": clean(test["mae"][s]),
                                       "baseline_mae": clean(test["baseline_mae"][s]),
                                       "gain_pct": clean(test["gain_pct"][s])} for s in test["mae"]}},
        "players": sorted(players, key=lambda p: p["player_name"]),
    }
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False))

    page = STATIC_PAGE.read_text().replace(
        "<title>", '<meta name="nba-data" content="data.json">\n<title>', 1)
    (out / "index.html").write_text(page)
    (out / ".nojekyll").write_text("")          # GitHub Pages : servir les fichiers tels quels
    print(f"{len(players)} joueurs exportés dans {out}/ (coût des nouvelles explications : {total_cost:.4f} €)")

    # Tableau prêt à coller dans la section "Results" du README.
    print("\n| Stat | Model MAE | Baseline MAE | Gain |\n|---|---:|---:|---:|")
    for s, m in data["model"]["test_metrics"].items():
        cell = lambda v, f, unit="": "—" if v is None else format(v, f) + unit
        print(f"| {s} | {cell(m['mae'], '.2f')} | {cell(m['baseline_mae'], '.2f')} | {cell(m['gain_pct'], '+.0f', ' %')} |")


if __name__ == "__main__":
    main()
