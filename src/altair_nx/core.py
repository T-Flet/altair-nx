import pandas as pd
import numpy as np
from numpy.typing import NDArray

import networkx as nx
import altair as alt

from math import atan2, sin, cos, pi, sqrt
from itertools import chain

from ._utils import despine


def to_pandas_nodes(G: nx.Graph, pos: dict[..., NDArray[np.float_]]):
    '''Convert Graph nodes to pandas DataFrame meant for drawing with Altair.
    
    :param G: The graph to draw.
    :param pos: The node positions of G, as produced by any of the nx.*_layout functions, e.g. nx.kamada_kawai_layout, which is the default if pos is None.
    
    :return: A pandas DataFrame of nodes.
    '''
    assert not (overlap := set(chain.from_iterable(G.nodes[n].keys() for n in G.nodes)).intersection(avoid := ['node', 'x', 'y'])), f'nodes in G should not have attributes named any of {avoid}; overlapping attributes: {overlap}'
    return pd.DataFrame([dict(node = n, x = pos[n][0], y = pos[n][1], **G.nodes[n]) for n in G.nodes], index = G.nodes)



def to_pandas_edges(G: nx.Graph, pos: dict[..., NDArray[np.float_]], control_points: list[tuple[float, float]] = None,
    loop_radius = .03, loop_angle = 90., loop_n_points = 30):
    '''Convert Graph edges to pandas DataFrame meant for drawing with Altair.

    :param G: The graph to draw.
    :param pos: The node positions of G, as produced by any of the nx.*_layout functions, e.g. nx.kamada_kawai_layout, which is the default if pos is None.
    :param control_points: Points to insert in the dataframe between the source and target point rows of each edge;
        they should be expressed in coordinates relative to their straight edge:
        (proportion of edge length parallel to the edge, proportion of edge length perpendicular (anticlockwise) to the edge).
        E.g. [(.5, .1)] is a single control point halfway along the edge and .1 of its length to the left of it.

    :param loop_radius: The radius of self-loop edges.
    :param loop_angle: The direction (in degrees) in which self-loops are drawn from the node; above it by default.
    :param loop_n_points: Number of points (INCLUDING the node itself) to draw self-loop edges.
        The default value is high in order to approximate a circle in both straight (i.e. control_points is None) and curved edge drawing cases,
        HOWEVER, low values such as 2, 3, and 4 produce nice shapes pointed at the node in both cases
        (in the straight-edge case, respectively: a short segment, a triangle, and a square).
        NOTE: to draw straight edges but interpolation-curved loops (rather than by high manual point count), set control_points to [] rather than None
        (or really any list of points whose 2nd coordinate is 0).

    :return: A pandas DataFrame of edges.
    '''
    assert not (overlap := set(chain.from_iterable(G.edges[n].keys() for n in G.edges)).intersection(avoid := ['edge', 'order', 'source', 'target', 'pair', 'x', 'y'])), f'edges in G should not have attributes named any of {avoid}; overlapping attributes: {overlap}'
    loop_angle *= pi / 180

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

        if e[0] == e[1]:
            centre_x = pos[e[0]][0] + loop_radius * cos(loop_angle)
            centre_y = pos[e[0]][1] + loop_radius * sin(loop_angle)
            for n in range(1, loop_n_points): # Pairs of relative coordinates: (D proportion parallel to D, D proportion perpendicular to D)
                point_angle = (loop_angle - pi + n * 2 * pi / loop_n_points) % (2 * pi)
                rows.append(dict(
                    edge = i, order = order,
                    source = e[0], target = e[1], pair = e,
                    x = centre_x + loop_radius * cos(point_angle),
                    y = centre_y + loop_radius * sin(point_angle),
                    **G.edges[e]
                ))
                order += 1
        elif control_points:
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



def to_pandas_edge_arrows(G: nx.Graph, pos: dict[..., NDArray[np.float_]], length: float, length_is_relative = False, control_points: list[tuple[float, float]] = None):
    '''Convert Graph edge arrows to pandas DataFrame meant for drawing with Altair.

    Note that arrows are not drawn for self-loops since they would convey no extra information (and also to avoid unnecessary clutter).
    
    :param G: The graph to draw.
    :param pos: The node positions of G, as produced by any of the nx.*_layout functions, e.g. nx.kamada_kawai_layout, which is the default if pos is None.
    :param length: A relative (i.e. proportion of edge length) or absolute measure of arrow length (the interpretation is determined by length_is_relative).
    :param length_is_relative: Whether arrow_length should be interpreted as a proportion of its edge length instead of an absolute measure.
    :param control_points: Points to insert in the dataframe between the source and target point rows of each edge;
        they should be expressed in coordinates relative to their straight edge:
        (proportion of edge length parallel to the edge, proportion of edge length perpendicular (anticlockwise) to the edge).
        E.g. [(.5, .1)] is a single control point halfway along the edge and .1 of its length to the left of it.
    
    :return: A pandas DataFrame of edge arrows.
    '''
    assert not (overlap := set(chain.from_iterable(G.edges[n].keys() for n in G.edges)).intersection(avoid := ['edge', 'order', 'source', 'target', 'pair', 'x', 'y'])), f'edges in G should not have attributes named any of {avoid}; overlapping attributes: {overlap}'

    rows = []
    for i, e in enumerate(G.edges):
        if e[0] != e[1]: # Arrows convey no extra information in self-loops
            Dx = pos[e[1]][0] - pos[e[0]][0]
            Dy = pos[e[1]][1] - pos[e[0]][1]
            D = sqrt(Dx ** 2 + Dy ** 2)
            angle = atan2(Dy, Dx)

            if control_points: # Shift direction from the source point to the last control point
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
                x = pos[e[1]][0] - length * (D if length_is_relative else 1) * cos(angle),
                y = pos[e[1]][1] - length * (D if length_is_relative else 1) * sin(angle),
                **G.edges[e]
            ))

    return pd.DataFrame(rows)



def to_chart(G: nx.Graph, pos: dict[..., NDArray[np.float_]]):
    '''Construct a single Altair Chart for the given Graph and node positions.
    
    :param G: The graph to draw.
    :param pos: The node positions of G, as produced by any of the nx.*_layout functions, e.g. nx.kamada_kawai_layout, which is the default if pos is None.
    
    :return: An Altair chart with layers for G's edges and nodes.
    '''
    node_df = to_pandas_nodes(G, pos)
    node_layer = alt.Chart(node_df)

    edge_df = to_pandas_edges(G, pos)
    edge_layer = alt.Chart(edge_df)

    chart = alt.LayerChart(layer = (edge_layer, node_layer))

    return despine(chart)


