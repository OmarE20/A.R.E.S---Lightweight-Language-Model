# ARES task runner. On Windows, `make` may be unavailable; the equivalent raw
# commands are documented in the README. PY can be overridden, e.g. `make PY=py train`.
PY ?= python

.PHONY: install data train train-smoke generate test clean

install:
	$(PY) -m pip install -r requirements.txt

# Download the corpus (TinyShakespeare) without training.
data:
	$(PY) -c "from config import Config; from data import download_corpus; download_corpus(Config().data_dir)"

# Full training run with the default config.
train:
	$(PY) train.py

# Short smoke run: char tokenizer, 500 iters — proves the pipeline end to end.
train-smoke:
	$(PY) train.py --tokenizer char --max_iters 500 --eval_interval 100 --warmup_iters 50

# Generate from the latest checkpoint. Override PROMPT, e.g. `make generate PROMPT="ROMEO:"`.
PROMPT ?= "\n"
generate:
	$(PY) generate.py --prompt $(PROMPT) --max_new_tokens 300 --temperature 0.8

test:
	$(PY) -m pytest -q

clean:
	rm -rf checkpoints plots data __pycache__ tests/__pycache__ .pytest_cache
