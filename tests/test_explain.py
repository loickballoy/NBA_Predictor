"""Tests de /explain et de la démo statique, avec un faux client Claude (aucun appel réseau)."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main
from api.explain import Explainer, Explanation, build_facts, cost_eur, unknown_numbers
from api.store import ModelStore
from pipeline.storage import LocalStorage
from tests.test_api import bucket, production  # noqa: F401  (fixture partagée)


class FakeClaude:
    """Renvoie les réponses prévues, dans l'ordre, et compte les appels."""
    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        text = self.texts.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=1200, output_tokens=300))


def facts_for(bucket):
    preds = pd.read_parquet(bucket / production(bucket)["predictions"]).set_index("id_PLAYER_ID")
    pid = int(preds.index[0])
    return pid, build_facts(preds.loc[pid], "Test Player", pid)


def answer(facts, extra=""):
    p = facts["projection"]
    pts = str(p["points"]).replace(".", ",")
    return json.dumps({
        "resume": f"Le modèle projette {pts} points par match en {facts['saison_projetee']}{extra}.",
        "facteurs": [f"Il a joué {facts['saisons_passees'][-1]['matchs']:g} matchs la saison passée.",
                     "Sa production est stable sur les trois dernières saisons."],
        "prudence": "Le modèle ne connaît ni les blessures ni le contexte de l'équipe.",
    }, ensure_ascii=False)


# ---------- vérification des nombres ----------

def test_numbers_from_facts_are_accepted_and_invented_ones_rejected(bucket):
    _, facts = facts_for(bucket)
    ok = Explanation.model_validate(json.loads(answer(facts)))
    assert unknown_numbers(ok, facts) == []
    bad = Explanation.model_validate(json.loads(answer(facts, extra=", soit une hausse de 37 %")))
    assert 37.0 in unknown_numbers(bad, facts)


def test_season_labels_are_not_checked_as_numbers(bucket):
    _, facts = facts_for(bucket)
    expl = Explanation(resume="Sur les saisons 2023-24 et 2025-26, le joueur est resté stable.",
                       facteurs=["Stable.", "Régulier."], prudence="Projection prudente pour ce joueur.")
    assert unknown_numbers(expl, facts) == []


# ---------- appel, repli, cache ----------

def test_valid_answer_is_used_cached_and_costed(bucket, tmp_path):
    pid, facts = facts_for(bucket)
    client = FakeClaude(answer(facts))
    ex = Explainer(client, LocalStorage(str(tmp_path)), log=lambda *a, **k: None)

    first = ex.explain("v1", pid, facts)
    assert first.source == "claude" and client.calls == 1
    assert first.cost_eur == pytest.approx(cost_eur("claude-haiku-5-5", 1200, 300))

    second = Explainer(FakeClaude(), LocalStorage(str(tmp_path)), log=lambda *a, **k: None).explain("v1", pid, facts)
    assert second.source == "cache" and second.cost_eur == 0      # relu depuis le bucket, gratuit


def test_invented_number_is_retried_then_accepted(bucket, tmp_path):
    pid, facts = facts_for(bucket)
    client = FakeClaude(answer(facts, extra=", +37 %"), answer(facts))
    r = Explainer(client, LocalStorage(str(tmp_path)), log=lambda *a, **k: None).explain("v1", pid, facts)
    assert r.source == "claude" and r.attempts == 2 and client.calls == 2
    assert "37" in r.rejected[0]


def test_two_bad_answers_fall_back_without_ai_and_are_not_cached(bucket, tmp_path):
    pid, facts = facts_for(bucket)
    client = FakeClaude("pas du JSON", answer(facts, extra=" et 99 rebonds"))
    logs = []
    r = Explainer(client, LocalStorage(str(tmp_path)), log=lambda *a, **k: logs.append(k)).explain("v1", pid, facts)
    assert r.source == "fallback"
    assert unknown_numbers(r.explanation, facts) == []            # le repli n'invente rien non plus
    assert not (tmp_path / f"explanations/v1/{pid}.json").exists()
    calls = [l for l in logs if l.get("event") == "llm_call"]
    assert len(calls) == 2 and all("cost_eur" in c and "input_tokens" in c for c in calls)


# ---------- endpoint ----------

def test_explain_endpoint(bucket, monkeypatch):
    pid, facts = facts_for(bucket)
    store = ModelStore(f"file://{bucket}", refresh_seconds=0)
    monkeypatch.setattr(api.main, "store", store)
    monkeypatch.setattr(api.main, "_explainer", Explainer(FakeClaude(answer(facts)), store.storage,
                                                          log=lambda *a, **k: None))
    body = TestClient(api.main.app).get(f"/players/{pid}/explain").json()
    assert body["source"] == "claude" and body["cost_eur"] > 0
    assert body["explanation"]["resume"].startswith("Le modèle projette")


# ---------- démo statique ----------

def test_demo_export_without_api_key(bucket, tmp_path, monkeypatch):
    from demo import export
    monkeypatch.setenv("STORAGE_URI", f"file://{bucket}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["export", "--out", str(tmp_path / "docs"), "--limit", "5"])
    export.main()

    data = json.loads((tmp_path / "docs/data.json").read_text())
    assert len(data["players"]) == 5
    assert all(p["season"] == "2026-27" and p["explanation"]["resume"] for p in data["players"])
    page = (tmp_path / "docs/index.html").read_text()
    assert '<meta name="nba-data" content="data.json">' in page
