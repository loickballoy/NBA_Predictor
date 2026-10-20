"""
Jeu d'évaluation de /explain : `python -m api.eval_explain --n 20`

Prend N joueurs du modèle en production (tirage fixe, reproductible), demande une explication
fraîche à Claude pour chacun (sans cache) et vérifie automatiquement :
  - que la réponse respecte le schéma JSON,
  - que chaque nombre cité existe dans les données envoyées,
  - qu'elle ne mentionne pas de faits que Claude ne peut pas connaître (blessure, transfert...).
Affiche un rapport et écrit le détail dans eval_explain.json. Nécessite ANTHROPIC_API_KEY.
"""
import argparse
import json
import os
import random
import sys

import anthropic

from api.explain import Explainer, build_facts
from api.store import ModelStore
from nba_api.stats.static import players as nba_players

FORBIDDEN = ["blessure", "blessé", "transfert", "échange", "contrat", "trade", "opération"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "claude-haiku-5-5"))
    parser.add_argument("--out", default="eval_explain.json")
    args = parser.parse_args()

    store = ModelStore(os.environ["STORAGE_URI"])
    snap = store.get()
    ids = sorted(int(i) for i in snap.predictions.index)
    sample = random.Random(args.seed).sample(ids, min(args.n, len(ids)))

    calls = []
    explainer = Explainer(anthropic.Anthropic(), store.storage, log=lambda *a, **f: calls.append(f),
                          model=args.model)
    rows = []
    for pid in sample:
        info = nba_players.find_player_by_id(pid)
        facts = build_facts(snap.predictions.loc[pid], info["full_name"] if info else None, pid)
        r = explainer.explain(snap.version, pid, facts, use_cache=False)
        text = " ".join([r.explanation.resume, *r.explanation.facteurs, r.explanation.prudence]).lower()
        rows.append({
            "player_id": pid, "player": facts["joueur"], "source": r.source, "attempts": r.attempts,
            "first_try_ok": r.source == "claude" and r.attempts == 1,
            "rejections": r.rejected, "forbidden_terms": [w for w in FORBIDDEN if w in text],
            "cost_eur": r.cost_eur, "explanation": r.explanation.model_dump(),
        })

    n = len(rows)
    report = {
        "model_version": snap.version, "llm_model": args.model, "n_players": n,
        "valid_first_try": sum(r["first_try_ok"] for r in rows),
        "valid_after_retry": sum(r["source"] == "claude" and r["attempts"] > 1 for r in rows),
        "fallback": sum(r["source"] == "fallback" for r in rows),
        "forbidden_terms": sum(bool(r["forbidden_terms"]) for r in rows),
        "total_cost_eur": round(sum(r["cost_eur"] for r in rows), 6),
        "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
    }
    with open(args.out, "w") as f:
        json.dump({"report": report, "players": rows}, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = report["fallback"] == 0 and report["forbidden_terms"] == 0
    print("\nÉVALUATION RÉUSSIE" if ok else "\nÉVALUATION À REVOIR : voir " + args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
