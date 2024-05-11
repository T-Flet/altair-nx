import pandas as pd
import networkx as nx
import altair as alt
from ._utils import despine
from math import atan2, sin, cos, sqrt


def to_pandas_nodes(G, pos):
    '''Convert Graph nodes to pandas DataFrame that's readable to Altair.
    '''
    return pd.DataFrame([dict(x = pos[n][0], y = pos[n][1], **G.nodes[n]) for n in G.nodes], index = G.nodes)



def to_pandas_edges(G, pos, control_points = None, **kwargs):
    '''Convert Graph edges to pandas DataFrame that's readable to Altair.

    control_points : list[tuple[float, float]]
        Points to insert in the dataframe between the source and target point rows of each edge; they should be expressed in coordinates relative to their straight edge:
        (proportion of edge length parallel to the edge, proportion of edge length perpendicular (anticlockwise) to the edge).
        E.g. [(0.5, 0.1)] is a single control point halfway along the edge and 0.1 of its length to the left of it.
    '''
    rows = []
    for i, e in enumerate(G.edges):
        Dx = pos[e[1]][0] - pos[e[0]][0]
        Dy = pos[e[1]][1] - pos[e[0]][1]
        D = sqrt(Dx ** 2 + Dy ** 2)
        angle = atan2(Dy, Dx)
        order = 0

        rows.append(dict(
            edge = i, order = order,
            source = e[0], target = e[1], pair = e,
            x = pos[e[0]][0], y = pos[e[0]][1],
            **G.edges[e]
        ))
        order += 1

        if control_points is not None:
            for v, w in control_points: # Pairs of relative coordinates: (D proportion parallel to D, D proportion perpendicular to D)
                rows.append(dict(
                    edge = i, order = order,
                    source = e[0], target = e[1], pair = e,
                    x = pos[e[0]][0] + D * (v * cos(angle) - w * sin(angle)),
                    y = pos[e[0]][1] + D * (v * sin(angle) + w * cos(angle)),
                    **G.edges[e]
                ))
                order += 1

        rows.append(dict(
            edge = i, order = order,
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0], y = pos[e[1]][1],
            **G.edges[e]
        ))

    return pd.DataFrame(rows)



def to_pandas_edges_arrows(G, pos, arrow_length, arrow_length_is_relative = False, control_points = None, **kwargs):
    '''Convert Graph edges to pandas DataFrame that's readable to Altair.
    If arrow_length_is_relative, then arrow_length is interpreted as the proportion of the corresponding edge length
    (it is interpreted as a proportion of the full length even if the edge is curved and the arrow angle is therefore based on the last control point).
    
    control_points : list[tuple[float, float]]
        Same argument as to_pandas_edges.
        NOTE that there is no curved_edges argument; it is assumed that if control_points is not None then the edges are indeed curved.
        Only the last control point is used for arrow purposes.
    '''
    rows = []
    for i, e in enumerate(G.edges):
        Dx = pos[e[1]][0] - pos[e[0]][0]
        Dy = pos[e[1]][1] - pos[e[0]][1]
        D = sqrt(Dx ** 2 + Dy ** 2)
        angle = atan2(Dy, Dx)

        if control_points is not None: # Shift direction from the source point to the last control point
            v, w = control_points[-1]
            Dx = pos[e[1]][0] - (pos[e[0]][0] + D * (v * cos(angle) - w * sin(angle)))
            Dy = pos[e[1]][1] - (pos[e[0]][1] + D * (v * sin(angle) + w * cos(angle)))
            angle = atan2(Dy, Dx)

        rows.append(dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0], y = pos[e[1]][1],
            **G.edges[e]
        ))

        rows.append(dict(
            edge = i, # Yes, if arrow_length_is_relative Dx & Dy are reassembled from D and cos & sin; this is to keep the same expression regardless of control_points
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0] - arrow_length * (D if arrow_length_is_relative else 1) * cos(angle),
            y = pos[e[1]][1] - arrow_length * (D if arrow_length_is_relative else 1) * sin(angle),
            **G.edges[e]
        ))

    return pd.DataFrame(rows)



def to_chart(G, pos):
    '''Construct a single Altair Chart for the given Graph and node positions.
    '''
    node_df = to_pandas_nodes(G, pos)
    node_layer = alt.Chart(node_df)

    edge_df = to_pandas_edges(G, pos)
    edge_layer = alt.Chart(edge_df)

    chart = alt.LayerChart(layer = (edge_layer, node_layer))

    return despine(chart)


