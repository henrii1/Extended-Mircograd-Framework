install:
	sudo apt update
	which pipx || sudo apt install -y pipx
	pipx ensurepath
	pipx list | grep poetry || pipx install poetry
	poetry install --no-root

env:
	poetry shell

test:
	python -m pytest -vv tests/test_*.py

format:
	black *.py

lint:
	pylint --disable=R,C,E1120 *.py

all: install lint test
 