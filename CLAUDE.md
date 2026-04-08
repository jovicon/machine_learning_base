# CLAUDE.md

Project context and conventions for Claude Code sessions.

## Project Purpose

ML study repo for working through **Andrew Ng's Machine Learning Specialization** (DeepLearning.AI / Coursera), with personal practice notebooks applying course concepts to real datasets.

## Structure

```
courses/machine_learning_specialization/course_1/
  week_1/   # Python/Jupyter basics, model representation, cost function, gradient descent
  week_2/   # NumPy vectorization, multiple variable linear regression
practices/
  linear_regression/        # Linear regression practice (English Premier League - Prac01)
  XGBClassifier/            # XGBoost classifier practice (Premier League - Prac02, result model)
real_models/
  premier_league/           # Production-ready Premier League model
    premier_league_train.ipynb   # Training notebook
    premier_league_app.ipynb     # App/inference notebook
    predicciones.json            # Latest predictions
    versiones/                   # Saved model versions (.pkl files)
```

## Environment Setup

```bash
make setup              # Create .venv, install deps, register Jupyter kernel
source .venv/bin/activate
make clean              # Remove .venv
```

Kernel name: `"Python (venv)"` — select this in VS Code when opening notebooks.

## Dependencies

Managed via `requirements.txt`. To add a new package:

```bash
source .venv/bin/activate
pip install <package>
pip freeze > requirements.txt
```

## Conventions

- Use **numpy vectorized operations** (no explicit Python loops) — consistent with course style
- Notebooks follow the course naming pattern: `C{course}_{week}_{lab}_{description}.ipynb`
- Practice notebooks live under `practices/` and apply course concepts to real-world data
- `.venv/`, `.env` are git-ignored — never commit them

## real_models Conventions

`real_models/` contains production-ready models (not practice notebooks).

### Model versioning (`versiones/`)

| File | Purpose |
|------|---------|
| `modelo_premier.pkl` | Active model — always the latest trained version |
| `modelo_premier_base.pkl` | Base reference model for comparison |
| `modelo_{YYYY-MM-DD}.pkl` | Dated snapshot (e.g., `modelo_2026-04-06.pkl`) |

Models are serialized with `joblib.dump()` as a dict containing:
`model_v3`, `feature_cols_v3`, `le`, `df`, `accuracy`, `accuracy_alta_confianza`, `fecha_entrenamiento`

### Workflow

1. **Train**: run `premier_league_train.ipynb` (all cells) → generates `versiones/modelo_{date}.pkl` + overwrites `modelo_premier.pkl`
2. **Predict**: run `premier_league_app.ipynb` → loads `modelo_premier.pkl`, generates predictions, saves to `predicciones.json`
3. **Retrain**: run cell 9 of train notebook every ~3 matchdays (~30 new matches); auto-compares accuracy vs previous model, only replaces if degradation ≤ 2%

### Prediction threshold

Only use predictions with **confidence > 55%** (marked `← USAR`). The model discards ~70% of matches where confidence is insufficient.

## Key Files

- `requirements.txt` — pinned Python dependencies
- `Makefile` — `setup`, `clean`, `requirements.txt` targets
- `courses/` — course lab solutions (source of truth for techniques)
- `practices/` — personal applied notebooks
