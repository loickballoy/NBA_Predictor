# NBA Home Predictor

I tried to have an AI predict the impossible: sports game outcomes.
This project trains a model to predict whether the home NBA team will win a matchup.

---

## 🔥 Results -- Progression

| Metric                   | Baseline | Current |
| ------------------------ | :------: | :-----: |
| **Accuracy**             |    49%   | **54%** |
| **Precision (Home win)** |    51%   | **58%** |

Goal was to increase precision (be right more often when we do call a home win), accepting some recall tradeoff.

---

## 🧠 What's inside

- **Feature engineering** : rolling eFG%, TOV%, ORB%, FTr and point margin over 3/5/10 games for each team across **all** games, then ***home − away*** deltas.
- **Context** features: day-of-week, team codes, Field Goal %, Free throws made, Points made, Poins taken, Wins and Losses.
- **Model**: RandomForest 


---

## ⚙️ Quickstart

### 1. Create env & install

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn joblib gradio pyarrow
```
### 2. Train & save model (from the notebook)

- Train the RandomForest on the engineered features.

## ⚠️ Limitations / Things to consider

- No injuries/lineups/odds; purely performance + schedule-derived features.

- Early season predictions have less history (we handle with neutral fills; can be improved with prior season carryover).

- Playoffs differ from regular season; treat separately for best results.

## 👤 Author

Developed by Balloy Loïck – Computer Engineering student at EPITA (SSSE Major).

- Portfolio: [https://www.lballoy-portfolio.com]

- GitHub: [https://github.com/loickballoy]

- LinkedIn: [https://www.linkedin.com/in/loick-balloy-332708203/]