from __future__ import annotations

import networkx as nx

import altair_nx as anx


def test_draw_small_graph_compiles_to_spec():
    G = nx.path_graph(5)
    pos = nx.spring_layout(G, seed=42)
    spec = anx.draw_networkx(G, pos).to_dict()
    assert isinstance(spec, dict)
    assert 'layer' in spec or 'mark' in spec


def test_draw_directed_graph_with_loops_compiles_to_spec():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (1, 1)])
    pos = nx.circular_layout(G)
    spec = anx.draw_networkx(G, pos, curved_edges=True).to_dict()
    assert isinstance(spec, dict)
    assert 'layer' in spec or 'mark' in spec
