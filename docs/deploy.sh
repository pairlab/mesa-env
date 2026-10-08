uv run sphinx-build -M html docs docs/_build -E
uv run python docs/prepare_html.py
uv run python -m http.server 8000 --directory docs/_build/html
