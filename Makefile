UV ?= uv

# Synchronize the project and its runtime dependencies into uv's managed
# .venv. The project is installed editable by default.
.PHONY: install
install:
	$(UV) sync

# Include the test toolchain for local development.
.PHONY: localdev
localdev:
	$(UV) sync --extra tests


# Build both the sdist and wheel using uv's isolated build environment.
.PHONY: dist
dist: clean-dist
	$(UV) build

.PHONY: clean-dist
clean-dist:
	rm -rf dist build


# Run publishing tools in uv-managed, ephemeral environments.
.PHONY: upload-prod
upload-prod: dist
	$(UV) tool run twine upload --repository tdvutil --skip-existing dist/*

.PHONY: upload-test
upload-test: dist
	$(UV) tool run twine upload --repository tdvutil_test --skip-existing dist/*


# Supply the Read the Docs requirements without adding documentation tooling
# to the project's runtime or test environments.
.PHONY: docs
docs:
	$(UV) run --with-requirements docs/requirements.txt \
		sphinx-build -M html docs docs/_build

.PHONY: clean-docs
clean-docs:
	rm -rf docs/_build/* docs/api


.PHONY: test
test:
	$(UV) run --extra tests pytest


.PHONY: clean
clean: clean-docs clean-dist
	rm -rf */*.egg-info *.egg-info


# Helpful for debugging Make variables.
print-%: ;@echo $*=$($*)
