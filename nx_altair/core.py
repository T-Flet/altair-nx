import pandas as pd
import networkx as nx
import altair as alt
from ._utils import despine
from math import atan2, sin, cos


def to_pandas_nodes(G, pos):
    '''Convert Graph nodes to pandas DataFrame that's readable to Altair.
    '''
    return pd.DataFrame([dict(x = pos[n][0], y = pos[n][1], **G.nodes[n]) for n in G.nodes], index = G.nodes)



def to_pandas_edges(G, pos, **kwargs):
    '''Convert Graph edges to pandas DataFrame that's readable to Altair.
    '''
    rows = []
    for i, e in enumerate(G.edges):
        rows.append(dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[0]][0], y = pos[e[0]][1],
            **G.edges[e]
        ))

        rows.append(dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0], y = pos[e[1]][1],
            **G.edges[e]
        ))

    return pd.DataFrame(rows)



def to_pandas_edges_arrows(G, pos, arrow_length, arrow_length_is_relative = False, **kwargs):
    '''Convert Graph edges to pandas DataFrame that's readable to Altair.
    If arrow_length_is_relative, then arrow_length is interpreted as the proportion of the corresponding edge length.
    '''
    rows = []
    for i, e in enumerate(G.edges):
        Dx = pos[e[1]][0] - pos[e[0]][0]
        Dy = pos[e[1]][1] - pos[e[0]][1]
        angle = atan2(Dy, Dx)

        rows.append(dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0], y = pos[e[1]][1],
            **G.edges[e]
        ))

        rows.append(dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0] - arrow_length * (Dx if arrow_length_is_relative else cos(angle)),
            y = pos[e[1]][1] - arrow_length * (Dy if arrow_length_is_relative else sin(angle)),
            **G.edges[e]
        ))

    return pd.DataFrame(rows)



def to_chart(G, pos):
    '''Construct a single Altair Chart for
    '''
    node_df = to_pandas_nodes(G, pos)
    node_layer = alt.Chart(node_df)

    edge_df = to_pandas_edges(G, pos)
    edge_layer = alt.Chart(edge_df)

    chart = alt.LayerChart(layer = (edge_layer, node_layer))

    return despine(chart)


