from __future__ import annotations

import importlib.metadata

import altair_nx as m


def test_version():
    assert importlib.metadata.version('altair_nx') == m.__version__
