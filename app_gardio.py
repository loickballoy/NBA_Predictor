# app_gradio.py
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import pathlib

# Optional: SHAP for explanations (works well with RandomForest)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

BUNDLE_PATH = "rf_homewin.joblib"
FEAT_PATH = "features_for_app.parquet"

bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
predictors = bundle["predictors"]
label_name = bundle["label_name"]

if pathlib.Path("features_for_app.parquet").exists():
    df = pd.read_parquet("features_for_app.parquet")
else:
    df = pd.read_csv("features_for_app.csv", parse_dates=["game_date"])
df["game_date"] = pd.to_datetime(df["game_date"])

# Identify name/code columns
home_name_col = next((c for c in df.columns if c.startswith("team_name_home")), None)
away_name_col = next((c for c in df.columns if c.startswith("team_name_away")), None)

if home_name_col and away_name_col:
    TEAMS = sorted(set(df[home_name_col]) | set(df[away_name_col]))
else:
    # fallback to numeric codes if names not present
    TEAMS = sorted(set(df["team_code_home"]) | set(df["team_code_away"]))

# Prepare SHAP explainer once
explainer = None
if HAS_SHAP:
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None

def list_dates(home, away):
    """Return available dates for this matchup from the feature store."""
    if home_name_col and away_name_col:
        sub = df[(df[home_name_col]==home) & (df[away_name_col]==away)]
    else:
        # if selecting by numeric code
        sub = df[(df["team_code_home"]==home) & (df["team_code_away"]==away)]
    dates = sorted(sub["game_date"].dt.date.unique().tolist())
    return [str(d) for d in dates]

def predict(home, away, date_str, threshold=0.70):
    # Filter to the requested row
    if home_name_col and away_name_col:
        sub = df[(df[home_name_col]==home) & (df[away_name_col]==away)]
    else:
        # If using codes, inputs home/away should be integers
        home = int(home); away = int(away)
        sub = df[(df["team_code_home"]==home) & (df["team_code_away"]==away)]

    sub = sub[sub["game_date"].dt.date == pd.to_datetime(date_str).date()]
    if sub.empty:
        return (
            "No data",
            "I don't have this matchup/date in the demo dataset.",
            None
        )

    X = sub[predictors].iloc[[0]]
    proba = float(model.predict_proba(X)[:,1][0])
    pred_label = "✅ Home win" if proba >= float(threshold) else "❌ Not confident"

    # Build a brief “why”
    why_lines = []
    if HAS_SHAP and explainer is not None:
        # For RF binary, shap_values may be list [neg, pos]; take positive class
        sv = explainer.shap_values(X)
        if isinstance(sv, list) and len(sv) == 2:
            shap_row = sv[1][0]
        else:
            shap_row = sv[0] if np.ndim(sv)==1 else sv[0][0]

        # Top 3 positive contributors toward home win
        contrib = sorted(zip(predictors, shap_row), key=lambda t: t[1], reverse=True)
        for name, val in contrib[:3]:
            why_lines.append(f"• {name}: +{val:.3f}")
        # Also show the strongest headwind (top negative)
        neg = sorted(zip(predictors, shap_row), key=lambda t: t[1])[:1]
        if neg:
            why_lines.append(f"• Headwind — {neg[0][0]}: {neg[0][1]:+.3f}")
    else:
        # Fallback: use absolute z-scores as a proxy (quick & dirty)
        row = X.iloc[0]
        z = ((row - df[predictors].mean()) / (df[predictors].std() + 1e-9)).fillna(0.0)
        top = z.sort_values(ascending=False)[:3]
        for name, val in top.items():
            why_lines.append(f"• {name} unusually high (z≈{val:.2f})")

    why_text = f"**P(Home win)**: {proba:.2%}  \n**Decision** (τ={float(threshold):.2f}): {pred_label}\n\nReasons:\n" + "\n".join(why_lines)

    # If actual outcome available, append it
    actual_txt = ""
    if label_name in sub.columns:
        actual = int(sub[label_name].iloc[0])
        actual_txt = "\n\n**Actual outcome:** " + ("🏠 Home won" if actual==1 else "🚌 Home lost")
    return pred_label, why_text + actual_txt, proba

with gr.Blocks(title="NBA Home Win Predictor") as demo:
    gr.Markdown("# 🏀 NBA Home Win Predictor (RandomForest)")
    gr.Markdown("Pick a historical matchup and date from the dataset to get a probability and a short **why**.")

    with gr.Row():
        home_in = gr.Dropdown(choices=TEAMS, label="Home team")
        away_in = gr.Dropdown(choices=TEAMS, label="Away team")
    date_in = gr.Dropdown(choices=[], label="Game date")

    with gr.Row():
        threshold_in = gr.Slider(minimum=0.5, maximum=0.9, step=0.01, value=0.7, label="Decision threshold (higher → more precision)")
        btn = gr.Button("Predict")

    pred_out = gr.Label(label="Prediction")
    why_out = gr.Markdown()
    proba_out = gr.Number(label="P(Home win)")

    # When home/away change, refresh the dates available
    def _update_dates(home, away):
        return gr.update(choices=list_dates(home, away), value=None)

    home_in.change(_update_dates, [home_in, away_in], [date_in])
    away_in.change(_update_dates, [home_in, away_in], [date_in])

    btn.click(predict, [home_in, away_in, date_in, threshold_in], [pred_out, why_out, proba_out])

if __name__ == "__main__":
    demo.launch()