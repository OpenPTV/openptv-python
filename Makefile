PROJECT := openptv_python
CONDA := conda
CONDAFLAGS :=
UV := uv
UVFLAGS := --extra dev --upgrade
COV_REPORT := html
PYTHON ?= python

default: qa unit-tests type-check

qa:
	$(PYTHON) -m pre_commit run --all-files

unit-tests:
	$(PYTHON) -m pytest -vv --cov=. --cov-report=$(COV_REPORT) --doctest-glob="*.md" --doctest-glob="*.rst"

type-check:
	$(PYTHON) -m mypy .

env-update: uv-env-update

conda-env-update:
	$(CONDA) install -y -c conda-forge conda-merge
	$(CONDA) run conda-merge environment.yml ci/environment-ci.yml > ci/combined-environment-ci.yml
	$(CONDA) env update $(CONDAFLAGS) -f ci/combined-environment-ci.yml

uv-env-update:
	$(UV) sync $(UVFLAGS)

docker-build:
	docker build -t $(PROJECT) .

docker-run:
	docker run --rm -ti -v $(PWD):/srv $(PROJECT)

template-update:
	pre-commit run --all-files cruft -c .pre-commit-config-cruft.yaml

docs-build:
	cp README.md docs/. && $(PYTHON) docs/render_native_stress_demo_include.py && cd docs && rm -fr _api && make clean && make html

# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW
