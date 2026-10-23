from __future__ import annotations

import networkx as nx
import pytest
from pandas import Timestamp

from altair_nx.timelapse import stitch_frames


def test_stitch_growing_graphs_shapes_columns_and_edge_ids():
    G1 = nx.Graph()
    G1.add_node(0, duty = 7)
    G2 = nx.Graph([(0, 1), (1, 2)])
    nx.set_node_attributes(G2, 3, 'duty')
    nx.set_edge_attributes(G2, 1.5, 'weight')
    G3 = nx.Graph([(0, 1)])
    nx.set_node_attributes(G3, 5, 'duty')
    nx.set_edge_attributes(G3, 2.5, 'weight')

    pos = {n: (float(n), -float(n)) for n in range(3)}
    frameset = stitch_frames([G1, G2, G3], positions = pos, frames = [7, 3, 5])

    assert frameset.n_frames == 3
    assert frameset.arrows is None and not frameset.directed

    # Snapshots stitched in playback order sorted by frame value: frame 3 (G2, 3 nodes), then 5 (G3, 2 nodes), then 7 (G1, 1 node)
    assert list(frameset.frames['frame_index']) == [0, 1, 2]
    assert list(frameset.frames['frame']) == [3, 5, 7]
    assert frameset.nodes['frame_index'].value_counts().sort_index().tolist() == [3, 2, 1]
    duties = list(frameset.nodes['duty'])
    assert duties[:3] == [3, 3, 3] and duties[-1] == 7 # playback order: the 3-node graph first, the 1-node graph last

    # Frame and frame_index columns plus every per-snapshot attribute column and the geometry ones are all present
    assert list(frameset.nodes.columns) == ['frame_index', 'frame', 'node', 'x', 'y', 'duty']
    assert list(frameset.edges.columns) == ['frame_index', 'frame', 'edge', 'order', 'source', 'target', 'pair', 'x', 'y', 'weight']

    # Edge ids are globally unique across frames; each edge occupies consecutive and strictly increasing order rows
    assert sorted(frameset.edges['edge']) == [0, 0, 1, 1, 2, 2]
    # DataFrames are flat and indexed by position (plain integer index), to be sliced by columns
    edge_groups = [list(g['order']) for k, g in frameset.edges.groupby('edge')]
    assert all(g == list(range(len(g))) for g in edge_groups)

    # Static positions: a node's coordinates are identical across the frames it appears in
    coords = frameset.nodes[['node', 'x', 'y']].drop_duplicates()
    assert coords['node'].is_unique


def test_stitch_sorts_datetime_frames_into_playback_order():
    frames = [Timestamp('2024-03-01'), Timestamp('2024-01-01'), Timestamp('2024-02-01')]
    months = ('03', '01', '02') # aligned with the frames above so that the graph for node-01 gets the January frame
    graphs = [nx.Graph() for _ in months]
    for G, month in zip(graphs, months, strict = True):
        G.add_node(f'node-{month}', month = month)
    pos = {f'node-{m}': (float(i), 0.) for i, m in enumerate(months)}

    frameset = stitch_frames(graphs, positions = pos, frames = frames)

    assert list(frameset.frames['frame']) == sorted(frames)
    assert str(frameset.frames.dtypes['frame']).startswith('datetime64')
    assert str(frameset.nodes.dtypes['frame']).startswith('datetime64')
    assert list(frameset.nodes['node']) == ['node-01', 'node-02', 'node-03'] # playback order, NOT the given order
    pos = {f'node-{m}': (float(i), 0.) for i, m in enumerate(('01', '02', '03'))}

    frameset = stitch_frames(graphs, positions = pos, frames = frames)

    assert list(frameset.frames['frame']) == sorted(frames)
    assert str(frameset.frames.dtypes['frame']).startswith('datetime64')
    assert str(frameset.nodes.dtypes['frame']).startswith('datetime64')
    assert list(frameset.nodes['node']) == ['node-01', 'node-02', 'node-03'] # playback order, NOT the given order


def test_stitch_single_positions_dict_is_shared_by_all_frames():
    graphs = [nx.path_graph(3), nx.path_graph(2)]
    pos = {n: (.5 * n, .25 * n) for n in range(3)}
    frameset = stitch_frames(graphs, positions = pos)
    assert list(frameset.frames['frame']) == [0, 1]
    assert frameset.nodes.groupby('node')[['x', 'y']].nunique().max().max() == 1 # every node gets one shared position across all its frames


def test_stitch_default_union_layout_is_static_without_positions():
    graphs = [nx.path_graph(3), nx.path_graph(2)]
    frameset = stitch_frames(graphs) # positions defaults to one kamada_kawai layout over the union of the snapshots (requires scipy)
    assert frameset.nodes.groupby('node')[['x', 'y']].nunique().max().max() == 1


def test_stitch_bakes_geometry_of_loops_control_points():
    G1 = nx.Graph([(1, 1), (0, 1)])
    G2 = nx.Graph([(2, 3)])
    pos = {0: (0., 0.), 1: (1., 0.), 2: (0., 1.), 3: (1., 1.)}
    frameset = stitch_frames([G1, G2], positions = pos, frames = ['a', 'b'], control_points = [(.5, .2)])

    loop_rows = frameset.edges[frameset.edges['edge'] == 1] # edge id 1 is the (1, 1) self-loop of frame 'a' (edge ids of frame 'a' start at 0)
    assert len(loop_rows) > 2 # loop geometry requires more points than an edge's source and target
    straight_edge_rows = frameset.edges.loc[frameset.edges['pair'] == (2, 3)]
    assert len(straight_edge_rows) == 3 # source, control point and target rows, so the control point is baked in
    assert straight_edge_rows['order'].tolist() == [0, 1, 2]


def test_stitch_directed_sequences_build_arrows():
    D1 = nx.DiGraph([(0, 1)])
    D2 = nx.DiGraph([(0, 1), (1, 0)])
    D1.add_edge(1, 1) # self-loops produce edge rows but no arrow rows

    frameset = stitch_frames([D1, D2], positions = {0: (0., 0.), 1: (1., 0.)}, frames = [10, 20])

    assert frameset.directed and frameset.arrows is not None
    assert 'order' not in frameset.arrows.columns
    assert list(frameset.arrows['frame_index'].value_counts().sort_index()) == [2, 4] # one non-selfloop edge per frame 10; two in frame 20


def test_stitch_rejects_mixed_directedness():
    with pytest.raises(AssertionError, match = 'consistently'):
        stitch_frames([nx.Graph([(0, 1)]), nx.DiGraph([(0, 1)])], positions = {0: (0., 0.), 1: (1., 0.)}, frames = [0, 1])


def test_stitch_rejects_non_unique_frames_and_mismatched_counts():
    with pytest.raises(AssertionError, match = 'number of frame values'):
        stitch_frames([nx.Graph([(0, 1)])], positions = {0: (0., 0.), 1: (1., 0.)}, frames = [0, 1])
    with pytest.raises(AssertionError, match = 'unique'):
        stitch_frames([nx.Graph([(0, 1)]), nx.Graph([(0, 1)])], positions = {0: (0., 0.), 1: (1., 0.)}, frames = [3, 3])
    with pytest.raises(TypeError, match = 'orderable'):
        stitch_frames([nx.Graph([(0, 1)]), nx.Graph([(0, 1)])], positions = {0: (0., 0.), 1: (1., 0.)}, frames = [3, '4'])
    with pytest.raises(AssertionError, match = 'reserved'):
        G = nx.Graph([(0, 1)])
        nx.set_edge_attributes(G, 0, 'frame')
        stitch_frames([G, G], positions = {0: (0., 0.), 1: (1., 0.)}, frames = [0, 1])


def test_stitch_strictly_validates_matching_schemas_unconditionally():
    pos = {0: (0., 0.), 1: (1., 0.), 2: (2., 0.)}
    G1 = nx.Graph([(0, 1)]); nx.set_edge_attributes(G1, 1.0, 'weight')
    G_int = nx.Graph([(0, 1), (1, 2)]); nx.set_edge_attributes(G_int, 1, 'weight') # same column, different dtype
    G_missing = nx.Graph([(0, 1), (1, 2)]) # column missing entirely
    G_extra = nx.Graph([(0, 1), (1, 2)])
    nx.set_edge_attributes(G_extra, 0.5, 'weight'); nx.set_edge_attributes(G_extra, 'good', 'quality') # unexpected column

    with pytest.raises(ValueError, match = 'mismatched dtypes.*weight'):
        stitch_frames([G1, G_int], positions = pos, frames = [0, 1])
    with pytest.raises(ValueError, match = 'missing.*weight'):
        stitch_frames([G1, G_missing], positions = pos, frames = [0, 1])
    with pytest.raises(ValueError, match = 'unexpected.*quality'):
        stitch_frames([G1, G_extra], positions = pos, frames = [0, 1])
