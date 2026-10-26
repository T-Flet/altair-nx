'''
altair-nx: Draw interactive NetworkX graphs with Altair
'''

from ._version import __version__
from .draw_altair import draw_networkx, draw_networkx_edges, draw_networkx_nodes
from .timelapse import FrameSet, draw_networkx_timelapse, stitch_frames
from .util import copy_size_and_axes

__all__ = [
    'FrameSet',
    '__version__',
    'copy_size_and_axes',
    'draw_networkx',
    'draw_networkx_edges',
    'draw_networkx_nodes',
    'draw_networkx_timelapse',
    'stitch_frames',
]
