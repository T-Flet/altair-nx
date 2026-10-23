from __future__ import annotations

import networkx as nx
import pytest
from pandas import Timestamp

from altair_nx.timelapse import draw_networkx_timelapse, stitch_frames


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


def test_draw_timelapse_structure_slider_and_label_strip():
    pos = {0: (0., 0.), 1: (1., 0.), 2: (2., 0.)}
    frameset = stitch_frames([nx.path_graph(3), nx.path_graph(2)], positions = pos, frames = [Timestamp('2024-01-01'), Timestamp('2024-02-01')])
    viz = draw_networkx_timelapse(frameset, node_label = 'node')

    spec = viz.to_dict()
    assert len(spec['vconcat']) == 2 # the graph body and the frame-label strip
    layers = spec['vconcat'][0]['layer']
    assert [l['mark']['type'] for l in layers] == ['line', 'point', 'text'] # edges, nodes and labels; no arrows for undirected snapshots

    # Every layer filters to the selected snapshot, driven by one top-level range param named 'frame'
    assert all(l['transform'] == [{'filter': 'datum.frame_index == frame'}] for l in layers)
    assert spec['params'] == [{'name': 'frame', 'bind': {'input': 'range', 'min': 0, 'max': 1, 'step': 1, 'name': 'frame'}, 'value': 0}]

    strip = spec['vconcat'][1]
    assert strip['mark']['type'] == 'text'
    assert strip['encoding'] == {'text': {'field': 'label', 'type': 'nominal'}}
    assert list(viz.vconcat[1].data['label']) == ['frame: 2024-01-01', 'frame: 2024-02-01']
    assert strip['transform'] == [{'filter': 'datum.frame_index == frame'}]


def test_draw_timelapse_label_strip_position_toggle_and_nominal_int_frames():
    pos = {0: (0., 0.), 1: (1., 0.), 2: (2., 0.)}
    frameset = stitch_frames([nx.path_graph(3), nx.path_graph(2)], positions = pos)

    below = draw_networkx_timelapse(frameset)
    assert below.to_dict()['vconcat'][1]['mark']['type'] == 'text'
    assert below.to_dict()['vconcat'][1]['encoding']['text']['type'] == 'nominal'
    assert list(below.vconcat[1].data['label']) == ['frame: 0', 'frame: 1']

    above = draw_networkx_timelapse(frameset, frame_label_position = 'above')
    assert above.to_dict()['vconcat'][0]['mark']['type'] == 'text'

    with pytest.raises(TypeError, match = 'frame_label_position'):
        draw_networkx_timelapse(frameset, frame_label_position = 'side')


@pytest.mark.parametrize('frames, expected', [
    ([0, 1], ['frame: 0', 'frame: 1']),
    (['first', 'second'], ['frame: first', 'frame: second']),
    ([Timestamp('2024-01-01'), Timestamp('2024-02-01')], ['frame: 2024-01-01', 'frame: 2024-02-01']),
    ([Timestamp('2024-01-01 12:30:00')], ['frame: 2024-01-01 12:30:00']),
])
def test_draw_timelapse_stationary_named_labels(frames, expected):
    frameset = stitch_frames([nx.path_graph(2) for _ in frames], positions = {0: (0., 0.), 1: (1., 0.)}, frames = frames)
    original_frames = frameset.frames.copy()
    viz = draw_networkx_timelapse(frameset, frame_label_position = 'above', frame_label_font_size = 18, frame_label_colour = 'purple')

    strip = viz.to_dict()['vconcat'][0]
    assert strip['encoding'] == {'text': {'field': 'label', 'type': 'nominal'}}
    assert strip['mark'] == {'type': 'text', 'size': 18, 'color': 'purple'}
    assert strip['width'] == 500. and strip['height'] == 30
    assert strip['transform'] == [{'filter': 'datum.frame_index == frame'}]
    assert list(viz.vconcat[0].data['label']) == expected
    assert list(viz.vconcat[0].data.columns) == ['frame_index', 'frame', 'label']
    assert frameset.frames.equals(original_frames)


def test_draw_timelapse_filters_self_loops_orphans_and_subsets_per_frame():
    G = nx.Graph()
    G.add_edge('x', 'x')
    G.add_edge('x', 'y')
    G.add_node('iso')
    frameset = stitch_frames([G, G], positions = {'x': (0., 0.), 'y': (1., 0.), 'iso': (2., 0.)})

    viz = draw_networkx_timelapse(frameset, show_self_loops = False, show_orphans = False, node_subset = ['x'], edge_subset = [('x', 'y')])

    edge_layer, node_layer = viz.vconcat[0].layer[0], viz.vconcat[0].layer[1]
    assert all(edge_layer.data['source'] != edge_layer.data['target']) # the self-loop rows are gone
    assert list(edge_layer.data['pair']) == [('x', 'y')] * 4 # the drawn edge (source + control point rows) in both frames
    assert list(node_layer.data['node']) == ['x', 'x'] # orphan 'iso' and subset-dropped 'y' are gone in every frame


def test_draw_timelapse_draws_arrow_layers_only_for_directed_sequences():
    pos = {0: (0., 0.), 1: (1., 0.), 2: (2., 0.)}
    directed = stitch_frames([nx.DiGraph([(0, 1), (1, 0)]), nx.DiGraph([(0, 1)])], positions = pos)
    undirected = stitch_frames([nx.Graph([(0, 1), (1, 2)]), nx.Graph([(0, 1)])], positions = pos)

    assert [l['mark']['type'] for l in draw_networkx_timelapse(directed).to_dict()['vconcat'][0]['layer']] == ['line', 'line', 'point'] # edges, arrows, nodes
    assert [l['mark']['type'] for l in draw_networkx_timelapse(undirected).to_dict()['vconcat'][0]['layer']] == ['line', 'point'] # edges, nodes


def test_draw_timelapse_scales_coordinates_onto_union_domain():
    pos = {0: (0., 0.), 1: (2., 0.), 2: (2., 2.)}
    frameset = stitch_frames([nx.Graph([(0, 1)]), nx.Graph([(0, 1), (1, 2)])], positions = pos)
    viz = draw_networkx_timelapse(frameset, chart_width = 500., chart_height = 300.)

    nodes = viz.vconcat[0].layer[1].data
    frame0 = nodes[nodes['frame_index'] == 0].set_index('node')
    frame1 = nodes[nodes['frame_index'] == 1].set_index('node')
    # The same affine transform is applied to every frame: (x - min) / range stretched by 500/300 along y's longer x axis
    assert frame0.loc[0, 'x'] == pytest.approx(0.) and frame0.loc[0, 'y'] == pytest.approx(0.)
    assert frame0.loc[1, 'x'] == pytest.approx(5 / 3) and frame0.loc[1, 'y'] == pytest.approx(0.) # (2 - 0) / 2 * (500 / 300)
    assert frame1.loc[2, 'x'] == pytest.approx(5 / 3) and frame1.loc[2, 'y'] == pytest.approx(1.)

    # The padded axes' domains cover the drawn elements (the union of all frames' coordinates)
    x_domain, y_domain = viz.vconcat[0].encoding.x['scale']['domain'], viz.vconcat[0].encoding.y['scale']['domain']
    assert x_domain[0] < 0 < x_domain[1] > 5 / 3 and y_domain[0] < 0 < y_domain[1] > 1


def test_draw_timelapse_raises_on_empty_frameset_and_bad_arguments():
    with pytest.raises(ValueError, match = 'nodes or edges'):
        draw_networkx_timelapse(stitch_frames([nx.empty_graph(0)]))

    frameset = stitch_frames([nx.path_graph(2)], positions = {0: (0., 0.), 1: (1., 0.)})
    with pytest.raises(TypeError, match = 'node_subset'):
        draw_networkx_timelapse(frameset, node_subset = 'x')
    with pytest.raises(TypeError, match = 'edge_subset'):
        draw_networkx_timelapse(frameset, edge_subset = 'e')
