#################################################################################
# Sleeper-Agent Reproduction & CoT-Monitoring Probe Study
# Makefile in the spirit of cookiecutter-data-science.
#################################################################################

PROJECT_NAME = bucky
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

# Confirmed source dataset (CLAUDE.md 4.1)
DATASET_REPO_ID = lucywingard/anthropic-code-backdoor-train-data
RAW_DIR = data/raw
DATASET_DIR = $(RAW_DIR)/$(notdir $(DATASET_REPO_ID))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: create_environment check_environment requirements data data-force dashboard clean clean-data help

create_environment: ## Create the conda env from environment.yml
	conda env create -f environment.yml
	@echo ">>> Created conda env '$(PROJECT_NAME)'. Activate with: conda activate $(PROJECT_NAME)"

check_environment: ## Warn if the active conda env is not $(PROJECT_NAME)
	@if [ "$(CONDA_DEFAULT_ENV)" != "$(PROJECT_NAME)" ]; then \
		echo ">>> WARNING: active conda env is '$(CONDA_DEFAULT_ENV)', expected '$(PROJECT_NAME)'."; \
		echo ">>>          run: conda activate $(PROJECT_NAME)"; \
	fi

requirements: check_environment ## Install Python dependencies into the active env
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

data: check_environment $(DATASET_DIR) ## Download raw source data into data/raw/ (idempotent)

$(DATASET_DIR):
	$(PYTHON_INTERPRETER) scripts/download_data.py --repo-id $(DATASET_REPO_ID) --out $(RAW_DIR)

data-force: ## Force re-download of raw data even if present
	$(PYTHON_INTERPRETER) scripts/download_data.py --repo-id $(DATASET_REPO_ID) --out $(RAW_DIR)

dashboard: ## Launch the interactive Streamlit data/prompt dashboard
	$(PYTHON_INTERPRETER) -m streamlit run scripts/dashboard.py

clean: ## Delete compiled Python caches
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

clean-data: ## Delete downloaded raw data (keeps the directory + .gitkeep)
	rm -rf $(DATASET_DIR)

#################################################################################
# Self-documenting help                                                         #
#################################################################################

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
