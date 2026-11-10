.PHONY: help build wheel sdist sync develop test test-rust test-python lint fmt clean

# Toolchain configuration
UV ?= uv
CARGO ?= cargo
DIST_DIR ?= dist

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: wheel ## Alias for wheel

wheel: ## Build the alphax Python binary wheel (.whl) into dist/
	@mkdir -p $(DIST_DIR)
	$(UV) build --wheel crates/alphax-py --out-dir $(DIST_DIR)

sdist: ## Build the alphax Python source distribution (.tar.gz) into dist/
	@mkdir -p $(DIST_DIR)
	$(UV) build --sdist crates/alphax-py --out-dir $(DIST_DIR)

sync: ## Sync virtualenv and install alphax in editable mode
	$(UV) sync

develop: ## Build and install alphax into the active virtual environment in release mode
	$(UV) run --with "maturin>=1.9.4,<2.0" maturin develop --release -m crates/alphax-py/Cargo.toml

test: test-rust test-python ## Run both Rust and Python tests

test-rust: ## Run all Rust workspace tests
	$(CARGO) test --workspace

test-python: ## Run Python package tests
	$(UV) sync && $(UV) run --with pytest python -m pytest crates/alphax-py/tests -q

lint: ## Run clippy, cargo fmt check, and ruff lint
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	$(CARGO) fmt --check
	$(UV) run ruff check

fmt: ## Format Rust and Python code
	$(CARGO) fmt
	$(UV) run ruff format
	$(UV) run ruff check --fix

clean: ## Clean build artifacts, target wheels, and dist directory
	rm -rf $(DIST_DIR) crates/alphax-py/dist target/wheels
