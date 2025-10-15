help: ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

##############################################################################

lint: ## Run linters
	pyright src/ examples/ tools/

run_all_examples: ## Run all examples
	python3 tools/run_all_examples.py

check_expected_output: ## Check expected output of all examples
	python3 tools/check_expected_output.py

test: lint unittest run_all_examples check_expected_output ## Run all tests

unittest: ## Run unit tests
	python -m unittest discover -s tests -p "test_*.py"

unittest_verbose: ## Run unit tests in a verbose mode
	python -m unittest discover -s tests -p "test_*.py" -v