See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

This project is managed with [uv](https://docs.astral.sh/uv/),
which handles the Python version (see `.python-version`),
the project environment, and locked dependencies (`uv.lock`).
To set up a development environment with all dependency groups, run:

```console
$ uv sync --all-groups
```

Then run everything through `uv run`, which uses the project environment:

```console
$ uv run pytest # Python tests
$ uv run pre-commit run --all-files # Lint everything
$ uv run python -m sphinx -T -b html docs docs/_build/html # Build the docs
$ uv build # Make an SDist and wheel in dist/
```

# Pre-commit

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
uv tool install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```console
$ uv run pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```console
$ uv run pytest --cov
```

# Building docs

You can build the docs using:

```console
$ uv run python -m sphinx -T -b html docs docs/_build/html
```

You can see a preview with:

```console
$ uv run --with sphinx-autobuild sphinx-autobuild docs docs/_build/html --open-browser
```
