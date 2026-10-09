# RACRP — common tasks.
#   make setup      install the package and its dependencies (light)
#   make setup-all  install everything, including torch (~800 MB)
#   make test       run the test suite
#   make headline   reproduce the headline result (needs the trace)
#   make demo       serve the interactive demonstration
#   make docs       regenerate the report, deck and overview PDF
#   make verify     check every deliverable against the result tables

PY ?= python

.PHONY: help setup setup-all test headline results demo docs verify clean

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

setup:
	$(PY) -m pip install -e ".[demo,docs,dev]"

# Everything, including the deep-learning stack that reproduces the negative
# results. Nothing in the headline result needs this.
setup-all:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest

# The headline: the derived window against the deployed policy, and against a
# baseline strengthened to tune both of its percentiles. Needs the trace.
headline:
	$(PY) scripts/run_derived.py
	$(PY) scripts/run_strong_baseline.py

# Everything else that is reported, in dependency order. Hours, not minutes.
results:
	$(PY) scripts/run_characterise.py
	$(PY) scripts/run_baselines.py
	$(PY) scripts/run_information_audit.py
	$(PY) scripts/run_conditional.py
	$(PY) scripts/tune_smoothing.py
	$(PY) scripts/run_derived.py
	$(PY) scripts/run_strong_baseline.py
	$(PY) scripts/run_mechanism_check.py
	$(PY) scripts/run_sensitivity.py

demo:
	$(PY) -m racrp.web.server

docs:
	$(PY) scripts/finalize.py
	$(PY) scripts/make_overview_pdf.py

verify:
	$(PY) scripts/verify_deliverables.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
