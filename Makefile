.PHONY: setup activate clean

VENV_DIR = .venv
PYTHON = python3

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt
	$(VENV_DIR)/bin/python -m ipykernel install --user --name=venv --display-name "Python (venv)"
	@echo "✅ Done. Run: source .venv/bin/activate"

clean:
	rm -rf $(VENV_DIR)
	@echo "🗑️  venv removed"

requirements.txt:
	$(VENV_DIR)/bin/pip freeze > requirements.txt
	@echo "📦 requirements.txt updated"