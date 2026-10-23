from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any

import networkx as nx
import pandas as pd

from .core import to_pandas_edge_arrows, to_pandas_edges, to_pandas_nodes
from .util import validate_snapshot_schemas


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
        - a sequence of pos dicts (one per graph): per-frame positions, which is the way to make nodes move.
          TIP: to obtain smooth motion, lay each frame out starting from the previous frame's positions, e.g. `pos_next = nx.spring_layout(G_next, pos = pos_prev)`.
          Either way, all pos dicts must cover all the nodes ever appearing in the sequence; coherent motion and recognisability are the user's responsibility.
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
