"""
Explication des prédictions par Claude.

Garde-fous, du plus simple au plus important :
  1. Claude ne reçoit que des faits calculés par le pipeline (3 saisons, projection, écarts).
  2. Sa réponse doit être un JSON valide pour le schéma Pydantic `Explanation`.
  3. Chaque nombre cité doit exister dans les faits envoyés : sinon la réponse est rejetée.
  4. Après deux échecs, repli sur une explication générée sans IA, à partir des mêmes faits.

Chaque appel est journalisé avec ses tokens et son coût estimé en euros. Les explications
valides sont mises en cache dans le bucket, par version du modèle : jamais payées deux fois.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError

DEFAULT_MODEL = "claude-haiku-5-5"

# USD par million de tokens (entrée, sortie), prompts de moins de 100 000 tokens.
# Source : page tarifs Anthropic, octobre 2026. À vérifier si le modèle change.
PRICES_USD_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}

STATS = {
    "GP": "matchs",
    "MIN_PER_GAME": "minutes",
    "PTS_PER_GAME": "points",
    "REB_PER_GAME": "rebonds",
    "AST_PER_GAME": "passes",
    "STL_PER_GAME": "interceptions",
    "BLK_PER_GAME": "contres",
    "TOV_PER_GAME": "balles_perdues",
}

SYSTEM_PROMPT = """Tu es analyste NBA. On te donne les statistiques par match de trois saisons consécutives \
d'un joueur et la projection d'un modèle statistique pour la saison suivante.

Explique en français ce que prévoit le modèle et pourquoi c'est cohérent avec les données : âge, tendance \
sur les trois saisons, temps de jeu, matchs joués.

Règles strictes :
1. N'utilise QUE des nombres présents dans les données fournies, recopiés tels quels. N'invente aucun \
chiffre, aucun pourcentage, aucun calcul supplémentaire : les écarts utiles sont déjà fournis.
2. Ne mentionne aucun fait absent des données (blessure, transfert, équipe, contrat) : tu ne les connais pas.
3. Le modèle peut se tromper : indique sa principale limite pour ce joueur dans "prudence".

Réponds UNIQUEMENT par un objet JSON, sans texte autour :
{"resume": "2 ou 3 phrases", "facteurs": ["2 à 4 phrases courtes"], "prudence": "1 phrase"}"""


class Explanation(BaseModel):
    resume: str = Field(min_length=20, max_length=700)
    facteurs: list[str] = Field(min_length=2, max_length=4)
    prudence: str = Field(min_length=10, max_length=350)


@dataclass
class ExplainResult:
    explanation: Explanation
    source: str                     # "claude", "cache" ou "fallback"
    llm_model: Optional[str] = None
    cost_eur: float = 0.0           # coût de CETTE requête (0 si cache ou repli)
    attempts: int = 0
    rejected: list = field(default_factory=list)


# ---------- faits envoyés à Claude ----------

def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[2:]}"


def round_stat(stat: str, value: float) -> float:
    return float(round(value)) if stat == "GP" else round(float(value), 1)


def build_facts(row, player_name: Optional[str], player_id: int) -> dict:
    """row : une ligne du parquet de prédictions (colonnes x_s1_/x_s2_/x_s3_, <stat>_pred, <stat>_last)."""
    target = int(row["id_TARGET_YEAR"])
    seasons = []
    for k, prefix in enumerate(["x_s1_", "x_s2_", "x_s3_"]):
        season = {"saison": season_label(target - 3 + k)}
        for stat, label in STATS.items():
            col = prefix + stat
            if col in row and row[col] == row[col]:          # présent et non NaN
                season[label] = round_stat(stat, row[col])
        seasons.append(season)

    projection, ecart = {}, {}
    for stat, label in STATS.items():
        pred = round_stat(stat, row[f"{stat}_pred"])
        projection[label] = pred
        last = row.get(f"{stat}_last")
        if last is not None and last == last:
            ecart[label] = round_stat(stat, pred - round_stat(stat, last))

    facts = {
        "joueur": player_name or f"joueur {player_id}",
        "saison_projetee": season_label(target),
        "saisons_passees": seasons,
        "projection": projection,
        "ecart_projection_vs_derniere_saison": ecart,
    }
    age = row.get("x_s3_PLAYER_AGE")
    if age is not None and age == age:
        facts["age_derniere_saison"] = int(age)
    return facts


# ---------- vérification des nombres cités ----------

SEASON_RE = re.compile(r"\b(?:19|20)\d{2}\s*[-–/]\s*\d{2}\b")
NUMBER_RE = re.compile(r"(?<![\w.,])[-+−]?\d+(?:[.,]\d+)?")


def _numbers_in(value) -> list:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers_in(v)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers_in(v)]
    return []


def allowed_numbers(facts: dict) -> set:
    allowed = {1.0, 2.0, 3.0}                    # "trois saisons", "deux dernières saisons"...
    for n in _numbers_in(facts):
        allowed |= {n, abs(n), round(n), round(abs(n), 1)}
    return allowed


def cited_numbers(text: str) -> list:
    text = SEASON_RE.sub(" ", text)              # "2025-26" n'est pas un nombre à vérifier
    out = []
    for m in NUMBER_RE.findall(text):
        out.append(float(m.replace("−", "-").replace(",", ".")))
    return out


def unknown_numbers(expl: Explanation, facts: dict) -> list:
    """Nombres cités par Claude qui n'existent pas dans les faits (tolérance d'arrondi : 0,05)."""
    allowed = allowed_numbers(facts)
    text = " ".join([expl.resume, *expl.facteurs, expl.prudence])
    return [n for n in cited_numbers(text)
            if not any(abs(abs(n) - a) <= 0.051 for a in allowed)]


# ---------- repli sans IA ----------

def fallback_explanation(facts: dict, reason: str) -> Explanation:
    p, e = facts["projection"], facts["ecart_projection_vs_derniere_saison"]
    last = facts["saisons_passees"][-1]
    fmt = lambda v: f"{v:g}".replace(".", ",")
    biggest = sorted((k for k in e if k != "matchs"), key=lambda k: -abs(e[k]))[:2]
    return Explanation(
        resume=(f"Le modèle projette {fmt(p['points'])} points et {fmt(p['matchs'])} matchs pour "
                f"{facts['joueur']} en {facts['saison_projetee']}, contre {fmt(last.get('points', 0))} points "
                f"et {fmt(last.get('matchs', 0))} matchs la saison passée."),
        facteurs=[f"{k.replace('_', ' ').capitalize()} : {fmt(last.get(k, 0))} la saison passée, "
                  f"{fmt(p[k])} projeté." for k in biggest] or ["Projection proche de la saison passée.",
                                                               "Peu d'écart sur les statistiques."],
        prudence=f"Résumé automatique sans IA ({reason}).",
    )


# ---------- appel de Claude ----------

def cost_eur(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
    usd = input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
    return usd * float(os.environ.get("EUR_PER_USD", "0.86"))   # taux approximatif, configurable


def _parse(text: str) -> Explanation:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("aucun objet JSON dans la réponse")
    return Explanation.model_validate(json.loads(text[start:end + 1]))


class Explainer:
    def __init__(self, client, storage, log: Callable, model: str = DEFAULT_MODEL, max_attempts: int = 2):
        self.client = client            # anthropic.Anthropic(), ou un faux client dans les tests
        self.storage = storage
        self.log = log
        self.model = model
        self.max_attempts = max_attempts
        self._memory = {}

    @staticmethod
    def cache_key(version: str, player_id: int) -> str:
        return f"explanations/{version}/{player_id}.json"

    def cached(self, version: str, player_id: int) -> Optional[Explanation]:
        key = self.cache_key(version, player_id)
        if key in self._memory:
            return self._memory[key]
        data, _ = self.storage.read_json(key)
        if data is None:
            return None
        expl = Explanation.model_validate(data["explanation"])
        self._memory[key] = expl
        return expl

    def explain(self, version: str, player_id: int, facts: dict, use_cache: bool = True) -> ExplainResult:
        if use_cache:
            hit = self.cached(version, player_id)
            if hit is not None:
                self.log("INFO", "explication en cache", event="llm_call", cached=True, cost_eur=0.0,
                         player_id=player_id, model_version=version)
                return ExplainResult(hit, "cache")

        if self.client is None:
            return ExplainResult(fallback_explanation(facts, "clé API absente"), "fallback")

        messages = [{"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]
        result = ExplainResult(explanation=None, source="fallback", llm_model=self.model)
        for attempt in range(1, self.max_attempts + 1):
            result.attempts = attempt
            start = time.perf_counter()
            error, expl, text = None, None, ""
            usage_in = usage_out = 0
            try:
                resp = self.client.messages.create(model=self.model, max_tokens=1024,
                                                   system=SYSTEM_PROMPT, messages=messages)
                usage_in, usage_out = resp.usage.input_tokens, resp.usage.output_tokens
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                expl = _parse(text)
                bad = unknown_numbers(expl, facts)
                if bad:
                    error = f"nombres absents des données : {bad}"
            except (ValidationError, ValueError) as e:
                error = f"réponse invalide : {e}".splitlines()[0][:300]
            except Exception as e:   # noqa: BLE001 - réseau, quota, panne de l'API
                error = f"appel impossible : {type(e).__name__}"

            call_cost = cost_eur(self.model, usage_in, usage_out)
            result.cost_eur += call_cost
            self.log("INFO" if error is None else "WARNING", "appel LLM", event="llm_call", cached=False,
                     llm_model=self.model, input_tokens=usage_in, output_tokens=usage_out,
                     cost_eur=round(call_cost, 6), valid=error is None, error=error, attempt=attempt,
                     latency_ms=round((time.perf_counter() - start) * 1000), player_id=player_id,
                     model_version=version)

            if error is None:
                result.explanation, result.source = expl, "claude"
                self._store(version, player_id, expl)
                return result
            result.rejected.append(error)
            if text:   # on montre à Claude ce qui n'allait pas pour la tentative suivante
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Réponse refusée : {error}. Corrige en respectant "
                                                         "strictement les règles, JSON seul."}]

        result.explanation = fallback_explanation(facts, "réponse de l'IA rejetée")
        return result

    def _store(self, version: str, player_id: int, expl: Explanation) -> None:
        key = self.cache_key(version, player_id)
        self._memory[key] = expl
        try:
            # Création seulement : si une autre instance l'a écrit entre-temps, on garde la sienne.
            self.storage.write_json(key, {"explanation": expl.model_dump(), "llm_model": self.model},
                                    if_generation_match=None)
        except Exception as e:   # noqa: BLE001 - le cache est un bonus, pas une raison d'échouer
            self.log("WARNING", "écriture du cache impossible", error=type(e).__name__, key=key)
