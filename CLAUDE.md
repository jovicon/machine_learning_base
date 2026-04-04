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
  linear_regression/   # Personal practice notebooks (e.g., English Premier League)
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

## Key Files

- `requirements.txt` — pinned Python dependencies
- `Makefile` — `setup`, `clean`, `requirements.txt` targets
- `courses/` — course lab solutions (source of truth for techniques)
- `practices/` — personal applied notebooks
