# NBA Home Predictor

I tried to have an AI predict the impossible: sports game outcomes.
This project trains a model to predict whether the home NBA team will win, and serves a simple UI so anyone can try matchups and see a probability with a short “why”.

---

## 🔥 Results -- Progression

| Metric                   | Baseline | Current |
| ------------------------ | :------: | :-----: |
| **Accuracy**             |    40%   | **58%** |
| **Precision (Home win)** |    51%   | **61%** |

- Baseline: naive/poorly framed features.
- Current: leak-proof rolling features(Four Factors + rest), home–away deltas, time-aware splits, and tuned thresholding.

Goal was to increase precision (be right more often when we do call a home win), accepting some recall tradeoff.

---

## 🧠 What's inside

- **Feature engineering** (no leakage): rolling eFG%, TOV%, ORB%, FTr and point margin over 3/5/10 games for each team across **all** games, then ***home − away*** deltas.
- **Context** features: rest days, back-to-back flags, day-of-week, team codes.
- **Model**: RandomForest (saved with `joblib`), optionally a calibrated gradient-boosting baseline.
- **Demo app (Gradio)**: enter any home/away teams + a date; the app builds features from historical data before that date, predicts P(Home win), and shows a short rationale.

---

## ⚙️ Quickstart

### 1. Create env & install

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn joblib gradio pyarrow
```
### 2. Train & save model (from the notebook)

- Train the RandomForest on the engineered features.

- Save the artifact:
```python
import joblib
bundle = {"model": rf, "predictors": predictors, "label_name": "target_home", "team_categories": team_categories}
joblib.dump(bundle, "rf_homewin.joblib")
```

### 3. Run the demo app

```bash
python app_gradio.py
# Opens a local URL; pick Home team, Away team, and a date (YYYY-MM-DD)

```
---
## ⚠️ Limitations / Things to consider

- No injuries/lineups/odds; purely performance + schedule-derived features.

- Early season predictions have less history (we handle with neutral fills; can be improved with prior season carryover).

- Playoffs differ from regular season; treat separately for best results.

## 👤 Author

Developed by Balloy Loïck – Computer Engineering student at EPITA (SSSE Major).

- Portfolio: [https://www.lballoy-portfolio.com]

- GitHub: [https://github.com/loickballoy]

- LinkedIn: [https://www.linkedin.com/in/loick-balloy-332708203/]