import pandas as pd
import networkx as nx
import altair as alt
from ._utils import despine


def to_pandas_nodes(G, pos):
    '''Convert Graph nodes to pandas DataFrame that's readable to Altair.
    '''
    attributes = ['x', 'y']
    for n in G.nodes(): attributes += list(G.nodes[n].keys())
    attributes = list(set(attributes))

    df = pd.DataFrame(index = G.nodes(), columns = attributes)
    for n in G.nodes:
        data = dict(x = pos[n][0], y = pos[n][1], **G.nodes[n])
        df.loc[n] = data

    return df.infer_objects()



def to_pandas_edges(G, pos, **kwargs):
    '''Convert Graph edges to pandas DataFrame that's readable to Altair.
    '''
    attributes = ['source', 'target', 'x', 'y', 'edge', 'pair']
    if not (isinstance(G, nx.MultiGraph)):
        for e in G.edges(): attributes += list(G.edges[e].keys())
    attributes = list(set(attributes))

    df = pd.DataFrame(index = range(G.size()*2), columns = attributes)
    for i, e in enumerate(G.edges):
        idx = i * 2

        data1 = dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[0]][0], y = pos[e[0]][1],
            **G.edges[e]
        )

        data2 = dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0], y = pos[e[1]][1],
            **G.edges[e]
        )

        df.loc[idx] = data1
        df.loc[idx+1] = data2

    return df.infer_objects()



def to_pandas_edges_arrows(G, pos, arrow_length, **kwargs):
    '''Convert Graph edges to pandas DataFrame that's readable to Altair.
    '''
    attributes = ['source', 'target', 'x', 'y', 'edge', 'pair']
    for e in G.edges(): attributes += list(G.edges[e].keys())
    attributes = list(set(attributes))

    df = pd.DataFrame(index = range(G.size()*2), columns = attributes)

    for i, e in enumerate(G.edges):
        idx = i * 2
        Dx = pos[e[1]][0] - pos[e[0]][0]
        Dy = pos[e[1]][1] - pos[e[0]][1]

        data1 = dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0], y = pos[e[1]][1],
            **G.edges[e]
        )

        data2 = dict(
            edge = i,
            source = e[0], target = e[1], pair = e,
            x = pos[e[1]][0] - arrow_length * Dx, y = pos[e[1]][1] - arrow_length * Dy,
            **G.edges[e]
        )

        df.loc[idx] = data1
        df.loc[idx+1] = data2

    return df.infer_objects()



def to_chart(G, pos):
    '''Construct a single Altair Chart for
    '''
    node_df = to_pandas_nodes(G, pos)
    node_layer = alt.Chart(node_df)

    edge_df = to_pandas_edges(G, pos)
    edge_layer = alt.Chart(edge_df)

    chart = alt.LayerChart(layer = (edge_layer, node_layer))

    return despine(chart)


