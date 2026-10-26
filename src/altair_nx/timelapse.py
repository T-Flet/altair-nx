from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any

import altair as alt
import networkx as nx
import pandas as pd

from .core import to_pandas_edge_arrows, to_pandas_edges, to_pandas_nodes
from .draw_altair import draw_networkx_arrows, draw_networkx_edges, draw_networkx_labels, draw_networkx_nodes
from .util import apply_coordinate_affine, assemble_padded_layers, coordinate_affine, validate_snapshot_schemas


@dataclass
class FrameSet:
    '''Container of the DataFrames produced by stitching graph snapshots with `stitch_frames`.

    The DataFrames are flat long-form tables (plain integer index), meant to be sliced by their columns, e.g. `frameset.nodes.loc[frameset.nodes.frame_index == t]`.

    :ivar nodes: Long-form node DataFrame with columns 'frame_index', 'frame', 'node', 'x', 'y' plus the graphs' node attributes.
    :ivar edges: Long-form edge DataFrame with columns 'frame_index', 'frame', 'edge', 'order', 'source', 'target', 'pair', 'x', 'y' plus the graphs' edge attributes.
    :ivar arrows: Long-form arrow DataFrame (only present for directed snapshots) with the same columns as edges except for 'order'; None if the snapshots are undirected.
    :ivar frames: Side table with columns 'frame_index' (0..n-1 in playback order) and 'frame' (the original frame values, in sorted playback order), used for displaying frame labels/tooltips.
    :ivar directed: Whether the snapshots are directed.
    '''
    nodes: pd.DataFrame
    edges: pd.DataFrame
    frames: pd.DataFrame
    arrows: pd.DataFrame | None = None
    directed: bool = False

    @property
    def n_frames(self) -> int:
        '''Number of stitched snapshots.'''
        return len(self.frames)

    def __repr__(self) -> str:
        arrows = 'None (undirected)' if self.arrows is None else str(self.arrows.shape)
        return f'FrameSet({len(self.frames)} frames; nodes: {self.nodes.shape}, edges: {self.edges.shape}, arrows: {arrows})'


def stitch_frames(graphs: Sequence[nx.Graph], positions: dict[Any, tuple[float, float]] | Sequence[dict[Any, tuple[float, float]]] | None = None,
    frames: Sequence[Any] | None = None, *, control_points: list[tuple[float, float]] | None = None,
    loop_radius: float = .05, loop_angle: float = 90., loop_n_points: int = 30,
    arrow_length: float = .1, arrow_length_is_relative: bool = True) -> FrameSet:
    '''Stitch a sequence of graph snapshots into long-form DataFrames ready for drawing the whole sequence as a slider-driven timelapse with `draw_networkx_timelapse`.

    The snapshots are stitched at the DataFrame level: a nodes/edges(/arrows) DataFrame is constructed for each snapshot with the existing `to_pandas_*` converters
    (which bake all geometry, including self-loops and arrow shapes, into coordinates), then concatenated with two added columns:
    - 'frame_index': the snapshot's position (0..n-1) in playback order, i.e. sorted by frame value, used to select the frame in the drawing;
    - 'frame': the snapshot's given frame value (any orderable type, e.g. dates), kept for displaying frame labels/tooltips.

    The schemas of the snapshots must coincide: for each entity type (nodes; edges), all snapshots must carry the same attribute columns of the same dtypes;
    otherwise an error is raised listing the offending snapshots and instructing how to align their attributes. (This strictness is unconditional, since drifted attributes render incoherently anyway.)

    :param graphs: The snapshots in the sequence consistently directed or undirected.
    :param positions: Either
        - None (default): one layout for all frames, computed from the UNION of all snapshots' nodes and edges (a `nx.kamada_kawai_layout`, which requires scipy), so nodes stay static throughout the sequence instead of jumping between randomly-seeded per-frame layouts;
        - a single pos dict (as produced by the `nx.*_layout` functions): used for all frames;
        - a sequence of pos dicts (one per graph, in the same input order as graphs): per-frame positions, which is the way to make nodes move.
          TIP: to obtain smooth motion, lay each frame out starting from the previous frame's positions, e.g. `pos_next = nx.spring_layout(G_next, pos = pos_prev)`.
          Each pos dict must cover the nodes in its corresponding snapshot; a shared dict must cover all nodes in the sequence. Coherent motion and recognisability are the user's responsibility.
          Note: editing the returned DataFrames' 'x'/'y' columns is unsafe, since edge/arrow geometry is baked at stitch time from node coordinates.
    :param frames: One frame value per snapshot (unique and mutually orderable; defaults to a sequence of ints 0..n-1).
        Snapshots are stitched in playback order, a stable sort by their frame values, NOT necessarily in the order given.
    :param control_points: Points to place along the edges, expressed as a tuple of coordinates relative to their straight edge:
        (proportion of edge length parallel to the edge, proportion of edge length perpendicular (anticlockwise) to the edge).
        Geometry is fixed at stitch time: pass it here rather than in the drawing function (which only controls how it and its interpolation appear).
    :param loop_radius: The radius of self-loop edges (their geometry is fixed at stitch time).
    :param loop_angle: The direction (in degrees) in which self-loops are drawn from their node; above it by default.
    :param loop_n_points: Number of points (INCLUDING the node itself) used to draw self-loops.
    :param arrow_length: A relative (proportion of edge length; the default) or absolute arrow length; since arrow geometry is also fixed at stitch time,
        this argument lives here rather than in the drawing function (which retains all the other arrow_* styling arguments).
    :param arrow_length_is_relative: Whether arrow_length should be interpreted as a proportion of its edge length instead of an absolute measure.

    :return: A FrameSet of the long-form node, edge and (for directed snapshots) arrow DataFrames with 'frame_index'/'frame' columns,
        plus a side table of the frame values for display; ready to be drawn with `draw_networkx_timelapse`.
    '''
    assert graphs, 'at least one graph snapshot is required to stitch a timelapse.'
    directed = graphs[0].is_directed()
    assert len({G.is_directed() for G in graphs}) == 1, 'the graph snapshots must be consistently directed or undirected; a sequence mixing nx.DiGraph and nx.Graph cannot be stitched.'
    assert not (node_overlap := set(chain.from_iterable(G.nodes[n].keys() for G in graphs for n in G.nodes)).intersection(node_avoid := ['frame', 'frame_index'])), f'snapshot node attributes must not be named any of {node_avoid} (reserved for stitching frames); overlapping attributes: {node_overlap}'
    assert not (edge_overlap := set(chain.from_iterable(G.edges[n].keys() for G in graphs for n in G.edges)).intersection(edge_avoid := ['frame', 'frame_index'])), f'snapshot edge attributes must not be named any of {edge_avoid} (reserved for stitching frames); overlapping attributes: {edge_overlap}'

    if frames is None: frames = list(range(len(graphs)))
    frame_values = list(frames)
    assert len(frame_values) == len(graphs), f'the number of frame values ({len(frame_values)}) must match the number of graph snapshots ({len(graphs)}).'
    assert len(set(frame_values)) == len(frame_values), f'the frame values must be unique; duplicated: {sorted({f for f in frame_values if frame_values.count(f) > 1})}.'
    try: order = sorted(range(len(graphs)), key = lambda i: frame_values[i]) # stable sort of the snapshots into playback order
    except TypeError as exc: raise TypeError(f'the frame values must be mutually orderable, e.g. all numbers or all dates; {exc}') from exc
    frames, graphs_sorted = [frame_values[i] for i in order], [graphs[i] for i in order]

    if positions is None: all_pos = [nx.kamada_kawai_layout(nx.compose_all(graphs))] * len(graphs) # stability-first default: one layout shared by all frames
    elif isinstance(positions, dict): all_pos = [positions] * len(graphs)
    else:
        all_pos = list(positions)
        assert len(all_pos) == len(graphs) and all(isinstance(p, dict) for p in all_pos), 'positions must be None, a pos dict shared by all frames, or one pos dict per graph snapshot.'
        all_pos = [all_pos[i] for i in order]

    df_nodes: list[pd.DataFrame] = []
    df_edges: list[pd.DataFrame] = []
    df_arrows: list[pd.DataFrame] = []
    edge_offset = 0 # to keep edge ids globally unique across frames (their groups would otherwise span frames in the drawing)
    for frame_index, (frame, G_frame, pos) in enumerate(zip(frames, graphs_sorted, all_pos, strict = True)):
        nodes, edges = to_pandas_nodes(G_frame, pos), to_pandas_edges(G_frame, pos, control_points = control_points, loop_radius = loop_radius, loop_angle = loop_angle, loop_n_points = loop_n_points)
        arrows = to_pandas_edge_arrows(G_frame, pos, length = arrow_length, length_is_relative = arrow_length_is_relative, control_points = control_points) if directed else None
        if len(edges): edges['edge'] = edges['edge'] + edge_offset
        if arrows is not None and len(arrows): arrows['edge'] = arrows['edge'] + edge_offset
        edge_offset += len(G_frame.edges)
        nodes.insert(0, 'frame', frame); nodes.insert(0, 'frame_index', frame_index)
        edges.insert(0, 'frame', frame); edges.insert(0, 'frame_index', frame_index)
        if arrows is not None: arrows.insert(0, 'frame', frame); arrows.insert(0, 'frame_index', frame_index)
        df_nodes.append(nodes); df_edges.append(edges)
        if arrows is not None: df_arrows.append(arrows)

    validate_snapshot_schemas(df_nodes, frames, 'node')
    validate_snapshot_schemas(df_edges, frames, 'edge')
    if directed: validate_snapshot_schemas(df_arrows, frames, 'edge arrow')

    nodes = pd.concat(df_nodes, ignore_index = True)
    edges = pd.concat(df_edges, ignore_index = True)
    arrows = pd.concat(df_arrows, ignore_index = True) if directed else None

    frames_table = pd.DataFrame({'frame_index': range(len(frames)), 'frame': frames})
    return FrameSet(nodes = nodes, edges = edges, frames = frames_table, arrows = arrows, directed = directed)


def draw_networkx_timelapse(frame_set: FrameSet, *,
    node_subset: list[Any] | None = None, edge_subset: list[Any] | None = None, show_orphans: bool = True, show_self_loops: bool = True,
    node_size: int | str = 400, node_shape: str = 'circle', node_colour: str = 'teal', node_cmap: str | None = None, node_alpha: float | str | None = None,
    node_outline_width: float | str = 1., node_outline_dash_and_gap_lengths: tuple[float, float] | str | None = None, node_outline_colour: str | None = None,
    node_mark_kwargs: dict[str, Any] | None = None, node_encode_kwargs: dict[str, Any] | None = None,
    node_label: str | None = None, node_font_size: int | str = 15, node_font_colour: str = 'black', node_tooltip: list[str] | None = None, node_legend: bool = False,
    node_label_mark_kwargs: dict[str, Any] | None = None, node_label_encode_kwargs: dict[str, Any] | None = None,
    edge_width: int | str = 1, edge_dash_and_gap_lengths: tuple[float, float] | str | None = None, edge_colour: str = 'grey', edge_cmap: str | None = None, edge_alpha: float | str | None = None,
    edge_tooltip: list[str] | None = None, edge_legend: bool = False,
    curved_edges: bool = False, edge_interpolation: str = 'basis',
    edge_mark_kwargs: dict[str, Any] | None = None, edge_encode_kwargs: dict[str, Any] | None = None,
    arrow_width: int | str = 2, arrow_colour: str = 'black', arrow_cmap: str | None = None, arrow_alpha: float | str | None = None, arrow_legend: bool = False,
    arrow_mark_kwargs: dict[str, Any] | None = None, arrow_encode_kwargs: dict[str, Any] | None = None,
    chart_width: float | None = 500., chart_height: float | None = 300., chart_padding: float = .05,
    frame_label_font_size: int = 15, frame_label_colour: str = 'black', frame_label_position: str = 'below', frame_param_name: str = 'frame') -> alt.VConcatChart:
    '''Draw the frames of a FrameSet (as produced by `stitch_frames`) as a single interactive timelapse: a slider selects the position in the sequence and each view shows exactly that position's snapshot.

    The graph at slider position t is laid out over the whole sequence's data (using the union of all frames' node coordinates as the domain, which is why nodes stay put between frames
    even for growing graphs), while a per-layer snapshot filter (`datum.frame_index == frame`) limits every visual entity to the selected frame.
    The slider is a Vega-Lite range param (named by frame_param_name, 'frame' by default) declared once on the returned chart, so a single widget drives both the graph and the frame-label strip;
    users can reference this param by that name in their own `*_kwargs` encodings to add further interactivity (e.g. conditional fading of the previous frame).

    Curve GEOMETRY is a stitch-time concern: the edge/arrow/control-point shapes (and arrow lengths) are already baked into the FrameSet's DataFrames at stitch time.
    This function's curved_edges only controls how that geometry is INTERPOLATED into paths (edge_interpolation); it has no effect on the geometry itself.

    Note that the stitched data holds every frame at once (frames x entities rows) and can therefore exceed Altair's default 5000-row compile cap with large graphs;
    if so, call `alt.data_transformers.disable_max_rows()` first, or compose the output if fine-grained control is needed (the library does not disable the cap silently).

    Most styling arguments are identical to `draw_networkx`'s same-named arguments (defaults and semantics included) and are not re-documented here;
    see `draw_networkx` for tooltip/legend behaviour, attribute-driven styling and the `*_kwargs` injections (which name their layers, e.g. node_mark_kwargs).
    The differences arising from the timelapse flow are: `show_orphans` and the `node_/edge_subsets` act per frame on the stitched rows (a node is an orphan in the frames where it has no edges, not overall),
    curve/loop/arrow-geometry arguments live on `stitch_frames` instead, and `chart` (draw over a pre-existing chart) is not available.

    :param frame_set: The FrameSet to draw (as produced by `stitch_frames`); its DataFrames are copied, not modified.
    :param node_subset: Nodes (across all frames) to draw.
    :param edge_subset: Edges (across all frames, as (source, target) node pairs) to draw.
    :param show_orphans: Whether to draw, in each frame, that frame's nodes with no edges.
    :param show_self_loops: Whether to draw edges starting and ending on the same node;
        nodes with only self-loops will still be drawn (though edge-less in that frame) unless show_orphans is also False.

    :param curved_edges: Whether the baked edge geometry should be interpolated into smooth paths (using edge_interpolation).
    :param edge_interpolation: Interpolation method for the interpolated edges;
        see `draw_networkx`'s edge_interpolation for the available values and their geometric behaviour.

    :param chart_width: Same as `draw_networkx`'s chart_width; the aspect-ratio reference range is the union of all frames' node coordinates.
    :param chart_height: Same as `draw_networkx`'s chart_height.
    :param chart_padding: Same as `draw_networkx`'s chart_padding.

    :param frame_label_font_size: Size of the stationary, centred frame-label strip's text (the param name followed by the current frame's original value, e.g. 'time: 2024-01-01').
    :param frame_label_colour: Colour of the frame-label strip's text.
    :param frame_label_position: Whether the frame-label strip appears above or below the graph.
    :param frame_param_name: Name of the Vega-Lite range param that drives the slider, also used as the displayed label next to it and the prefix in the frame-label strip;
        change it to rename that word (e.g. 'frame' to 'time') and to reference the param from your own `*_kwargs` encodings under the new name.

    :return: A vertically concatenated Altair chart (alt.VConcatChart): the graph view (whose possible layers, in order, are [edges, arrows, nodes, labels];
        arrows are present only for directed snapshots and labels only if node_label is not None) and the frame-label strip,
        ordered by frame_label_position. The graph view's size and axes' domains can be extracted with the same helpers as draw_networkx outputs
        (e.g. `output.vconcat[index].width` and `output.vconcat[index].encoding.x['scale']['domain']`).
    '''
    if node_subset is not None and not isinstance(node_subset, list): raise TypeError('node_subset must be a list or None.')
    if edge_subset is not None and not isinstance(edge_subset, list): raise TypeError('edge_subset must be a list or None.')
    if frame_label_position not in ('above', 'below'): raise TypeError('frame_label_position must be \'above\' or \'below\'.')
    if not frame_param_name.isidentifier(): raise ValueError('frame_param_name must be a valid identifier (it names the slider param, which is referenced inside the chart\'s filtering expressions).')

    df_nodes = frame_set.nodes.copy()
    df_edges = frame_set.edges.copy()
    df_arrows = frame_set.arrows.copy() if frame_set.arrows is not None else None


    # ---------- Determine the rows to keep (per-frame semantics) ------------

    node_keep = pd.Series(True, index = df_nodes.index)
    edge_keep = pd.Series(True, index = df_edges.index)
    if len(df_edges):
        if not show_self_loops: edge_keep &= df_edges['source'] != df_edges['target'] # False exactly on self-loop rows
        if edge_subset is not None: edge_keep &= df_edges['pair'].isin(edge_subset)
    if not show_orphans and len(df_nodes):
        if len(df_edges):
            active_edges = df_edges if show_self_loops else df_edges[df_edges['source'] != df_edges['target']]
            active_endpoints = set(zip(active_edges['frame_index'], active_edges['source'], strict = True)) | set(zip(active_edges['frame_index'], active_edges['target'], strict = True))
            node_keep = pd.Series(pd.MultiIndex.from_arrays([df_nodes['frame_index'], df_nodes['node']]).isin(active_endpoints), index = df_nodes.index)
        else: node_keep = pd.Series(False, index = df_nodes.index) # every node is an orphan when the sequence contains no edges
    if node_subset is not None: node_keep &= df_nodes['node'].isin(node_subset)

    df_nodes = df_nodes[node_keep]
    df_edges = df_edges[edge_keep]
    if df_arrows is not None and len(df_arrows):
        arrow_keep = pd.Series(True, index = df_arrows.index)
        if edge_subset is not None: arrow_keep &= df_arrows['pair'].isin(edge_subset)
        df_arrows = df_arrows[arrow_keep]


    if not (len(df_nodes) or len(df_edges)): raise ValueError('the FrameSet does not contain any nodes or edges.')


    # ---------- Scale all coordinates onto the union of ALL frames' node coordinates (row filtering must not affect the scale) ------------

    x_min, x_max = frame_set.nodes['x'].min(), frame_set.nodes['x'].max()
    y_min, y_max = frame_set.nodes['y'].min(), frame_set.nodes['y'].max()
    x_affine, y_affine, chart_width, chart_height = coordinate_affine(chart_width, chart_height, x_min, x_max - x_min, y_min, y_max - y_min)

    df_nodes['x'] = apply_coordinate_affine(df_nodes['x'].tolist(), x_affine)
    df_nodes['y'] = apply_coordinate_affine(df_nodes['y'].tolist(), y_affine)
    if len(df_edges):
        df_edges['x'] = apply_coordinate_affine(df_edges['x'].tolist(), x_affine)
        df_edges['y'] = apply_coordinate_affine(df_edges['y'].tolist(), y_affine)
    if df_arrows is not None and len(df_arrows):
        df_arrows['x'] = apply_coordinate_affine(df_arrows['x'].tolist(), x_affine)
        df_arrows['y'] = apply_coordinate_affine(df_arrows['y'].tolist(), y_affine)


    # ---------- Construct the layers ------------

    layers = []

    # Edges
    if len(df_edges):
        edges = draw_networkx_edges(layer = alt.Chart(df_edges),
            width = edge_width, dash_and_gap_lengths = edge_dash_and_gap_lengths, colour = edge_colour, cmap = edge_cmap, alpha = edge_alpha,
            tooltip = edge_tooltip, legend = edge_legend,
            curved_edges = curved_edges, interpolation = edge_interpolation,
            mark_kwargs = edge_mark_kwargs, encode_kwargs = edge_encode_kwargs)
        layers.append(edges)

        # Arrows
        if df_arrows is not None and len(df_arrows):
            arrows = draw_networkx_arrows(layer = alt.Chart(df_arrows),
                width = arrow_width, dash_and_gap_lengths = edge_dash_and_gap_lengths,
                colour = arrow_colour, cmap = arrow_cmap, alpha = arrow_alpha,
                tooltip = edge_tooltip, legend = arrow_legend,
                mark_kwargs = arrow_mark_kwargs, encode_kwargs = arrow_encode_kwargs)
            layers.append(arrows)

    # Nodes
    if len(df_nodes):
        nodes = draw_networkx_nodes(layer = alt.Chart(df_nodes),
            size = node_size, shape = node_shape,
            colour = node_colour, cmap = node_cmap, alpha = node_alpha,
            outline_width = node_outline_width, outline_dash_and_gap_lengths = node_outline_dash_and_gap_lengths, outline_colour = node_outline_colour,
            tooltip = node_tooltip, legend = node_legend,
            mark_kwargs = node_mark_kwargs, encode_kwargs = node_encode_kwargs)
        layers.append(nodes)

        # Node labels
        if node_label:
            labels = draw_networkx_labels(layer = alt.Chart(df_nodes),
                label = node_label, font_size = node_font_size, font_colour = node_font_colour,
                mark_kwargs = node_label_mark_kwargs, encode_kwargs = node_label_encode_kwargs)
            layers.append(labels)

    if not layers: raise ValueError('the FrameSet does not contain any nodes or edges.')


    # ---------- Add the snapshot filters and assemble the layers ------------

    frame_param = alt.param(name = frame_param_name, value = 0, bind = alt.binding_range(min = 0, max = frame_set.n_frames - 1, step = 1, name = frame_param_name))
    layers = [l.transform_filter(f'datum.frame_index == {frame_param_name}') for l in layers]

    body = assemble_padded_layers(layers, chart_width, chart_height, chart_padding)


    # ---------- Build the frame-label strip ------------

    df_frames = frame_set.frames.copy()
    df_frames['label'] = frame_param_name + ': ' + df_frames['frame'].astype(str)
    frame_label_chart = alt.Chart(df_frames).mark_text(size = frame_label_font_size, color = frame_label_colour).encode(
        alt.Text('label', type = 'nominal')
    ).transform_filter(f'datum.frame_index == {frame_param_name}').properties(width = chart_width, height = 30)


    # ---------- Concatenate the views and attach the slider ------------

    views = (frame_label_chart, body) if frame_label_position == 'above' else (body, frame_label_chart)
    return alt.vconcat(*views).add_params(frame_param)
