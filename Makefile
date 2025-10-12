help: ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

##############################################################################

lint: ## Run linters
	pyright src/ examples/ tools/

run_all_examples: ## Run all examples
	python3 tools/run_all_examples.py