install:
	@uv sync

run :
	@uv run -m src

clean:
	@rm -rf ./.venv
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "__mypycache__" -exec rm -rf {} +

lint:
	@flake8 ./src
	@mypy ./src --warn-return-any--warn-unused-ignores--ignore-missing-imports--disallow-untyped-defs--check-untyped-defs