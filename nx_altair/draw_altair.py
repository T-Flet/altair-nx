import numpy as np
import altair as alt
import networkx as nx
from copy import deepcopy

from .core import to_pandas_edges, to_pandas_edges_arrows, to_pandas_nodes
from ._utils import is_arraylike


def draw_networkx_edges(G = None, pos = None,
    chart = None, layer = None,
    edgelist = None, width = 1, alpha = 1.0, edge_color = 'black', edge_cmap = None,
    curved_edges = False, control_points = None, interpolate = 'basis',
    tooltip = None, legend = False, **kwargs):
    '''Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    chart : altair chart
        A chart to plot on instead of creating a new one

    edgelist : collection of edge tuples
       Draw only specified edges

    width : float, or array of floats
       Line width of edges

    edge_color : colour string or edge attribute name
       In the attribute case, its values need to be colour strings if edge_cmap is None and floats if not None.

    alpha : float
       The edge transparency

    edge_cmap : Matplotlib colormap
       Colormap for mapping intensities of edges; SILENTLY IGNORED unless edge_color is the name of an edge attribute containing floats

    curved_edges : bool
        Whether edges should be curved (using control_points and interpolate arguments)
    
    control_points : list[tuple[float, float]]
        The control points to use the interpolation method on; they should be expressed in coordinates relative to their straight edge:
        (proportion of edge length parallel to the edge, proportion of edge length perpendicular (anticlockwise) to the edge).
        E.g. the default value of [(0.5, 0.1)] is a single control point halfway along the edge and 0.1 of its length to the left of it.
    
    interpolate : str
        Interpolation method for curved edges (which are built on control points provided in edge-relative coordinates in curved_edges).
        The default interpolation is a cubic spline (i.e. 'basis').
        Interactive examples of possible values: https://altair-viz.github.io/user_guide/marks/line.html
        Corresponding descriptions: https://d3js.org/d3-shape/curve

        Recommendation:
        - 'basis', 'catmull-rom' and 'bundle' (and of course 'linear' and 'monotone') are best since they do not overshoot if control points are close to endpoints
        - the open and closed interpolation varieties are not appropriate for edges
        - the 'natural' cubic and 'cardinal' interpolations tend to overshoot for control points close to endpoints

    Returns
    -------
    viz: ``altair.Chart`` object
    '''
    if curved_edges and control_points is None: control_points = [(0.5, 0.1)]
    elif not curved_edges: control_points = None # Because to_pandas_edges relies only on it to know whether edges are supposed to be curved

    if chart is None:
        df_edges = to_pandas_edges(G, pos, control_points = control_points)
        edge_chart = alt.Chart(df_edges)
    else:
        df_edges = chart.layer[0].data
        edge_chart = chart.layer[0]

    marker_attrs, encoded_attrs = {}, {}


    # ---------- Handle arguments ------------

    # Restrict to a given subset
    if isinstance(edgelist, list):
        df_edges = edge_chart.data = df_edges.loc[df_edges['pair'].isin(edgelist)]
    elif edgelist is not None: raise TypeError('edgelist must be a list or None.')

    # Width
    if isinstance(width, str): encoded_attrs['strokeWidth'] = alt.Size(width, legend = None)
    elif isinstance(width, (float, int)): marker_attrs['strokeWidth'] = width
    else: raise TypeError('width must be a string or int.')

    # Colour
    if not isinstance(edge_color, str): raise TypeError('edge_color must be a string (either a colour or the name of an edge attribute containing colour strings or floats for a colour map).')
    elif edge_color in df_edges.columns:
        if edge_cmap is None: encoded_attrs['color'] = alt.Color(edge_color, legend = None)
        elif isinstance(edge_cmap, str):
            if df_edges.dtypes[edge_color] == 'O': raise TypeError(f'the edge attribute ({edge_color}) to use with edge_cmap {edge_cmap} is non-numeric.')
            else: encoded_attrs['color'] = alt.Color(edge_color, scale = alt.Scale(scheme = edge_cmap), legend = None)
        else: raise TypeError('edge_cmap must be a string (colormap name) or None.')
    else: marker_attrs['color'] = edge_color

    # Opacity
    if isinstance(alpha, str): encoded_attrs['opacity'] = alpha
    elif isinstance(alpha, (int, float)): marker_attrs['opacity'] = alpha
    elif alpha is not None: raise TypeError('alpha must be a string or None.')

    # Curved edges
    if curved_edges:
        if isinstance(interpolate, str): marker_attrs['interpolate'] = interpolate
        else: raise TypeError('interpolate must be a string.')
    
    # Tooltip
    if tooltip is not None: encoded_attrs['tooltip'] = tooltip


    # ---------- Construct visualization ------------

    edge_chart = edge_chart.mark_line(**marker_attrs).encode(
        x = alt.X('x', axis = alt.Axis(title = '', grid = False, labels = False, ticks = False)),
        y = alt.Y('y', axis = alt.Axis(title = '', grid = False, labels = False, ticks = False)),
        detail = 'edge', order = 'order', **encoded_attrs
    )

    if chart is not None: chart.layer[0] = edge_chart

    return edge_chart



def draw_networkx_arrows(G = None, pos = None,
    chart = None, layer = None,
    edgelist = None, arrow_width = 2, arrow_length = 0.1, arrow_length_is_relative = True, alpha = 1.0, edge_color = 'black', edge_cmap = None,
    curved_edges = False, control_points = None,
    tooltip = None, legend = False, **kwargs):
    '''Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    chart : altair chart
        A chart to plot on instead of creating a new one

    edgelist : collection of edge tuples
       Draw only specified edges

    arrow_width : float, optional
       The width of arrow portions of edges.

    arrow_length : float, optional
       The proportion of the line to be occupied by the arrow.
    
    arrow_length_is_relative: bool, optional
        Whether arrow_length should be interpreted as a proportion of its edge length instead of an absolute measure.

    edge_color : colour string or edge attribute name
       In the attribute case, its values need to be colour strings if edge_cmap is None and floats if not None.

    alpha : float
       The edge transparency

    edge_cmap : Matplotlib colormap
       Colormap for mapping intensities of edges; SILENTLY IGNORED unless edge_color is the name of an edge attribute containing floats
    
    curved_edges : bool
        Same argument as draw_networkx_edges: whether the edges to be arrow-ed are curved
    
    control_points : list[tuple[float, float]]
        Same argument as to_pandas_edges; only the last control point is used for arrow purposes

    Returns
    -------
    viz: ``altair.Chart`` object
    '''
    if curved_edges and control_points is None: control_points = [(0.5, 0.1)]
    elif not curved_edges: control_points = None # Because to_pandas_edges_arrows relies only on it to know whether edges are supposed to be curved

    if chart is None:
        df_edge_arrows = to_pandas_edges_arrows(G, pos, arrow_length, arrow_length_is_relative = arrow_length_is_relative, control_points = control_points)
        edge_chart = alt.Chart(df_edge_arrows)
    else:
        df_edge_arrows = chart.layer[0].data
        edge_chart = chart.layer[0]

    marker_attrs, encoded_attrs = {}, {}


    # ---------- Handle arguments ------------

    # Restrict to a given subset
    if isinstance(edgelist, list):
        df_edge_arrows = edge_chart.data = df_edge_arrows.loc[df_edge_arrows['pair'].isin(edgelist)]
    elif edgelist is not None: raise TypeError('edgelist must be a list or None.')

    # Width
    if isinstance(arrow_width, str): encoded_attrs['size'] = alt.Size(arrow_width, legend = None)
    elif isinstance(arrow_width, (float, int)): marker_attrs['strokeWidth'] = arrow_width
    else: raise TypeError('arrow_width must be a string or int.')

    # Colour
    if not isinstance(edge_color, str): raise TypeError('edge_color must be a string (either a colour or the name of an edge attribute containing colour strings or floats for a colour map).')
    elif edge_color in df_edge_arrows.columns:
        if edge_cmap is None: encoded_attrs['color'] = alt.Color(edge_color, legend = None)
        elif isinstance(edge_cmap, str):
            if df_edge_arrows.dtypes[edge_color] == 'O': raise TypeError(f'the edge attribute ({edge_color}) to use with edge_cmap {edge_cmap} is non-numeric.')
            else: encoded_attrs['color'] = alt.Color(edge_color, scale = alt.Scale(scheme = edge_cmap), legend = None)
        else: raise TypeError('edge_cmap must be a string (colormap name) or None.')
    else: marker_attrs['color'] = edge_color

    # Opacity
    if isinstance(alpha, str): encoded_attrs['opacity'] = alpha
    elif isinstance(alpha, (int, float)): marker_attrs['opacity'] = alpha
    elif alpha is not None: raise TypeError('alpha must be a string or None.')

    # Tooltip
    if tooltip is not None: encoded_attrs['tooltip'] = tooltip


    # ---------- Construct visualization ------------

    edge_chart = edge_chart.mark_line(**marker_attrs).encode(
        x = alt.X('x', axis = alt.Axis(grid = False, labels = False, ticks = False)),
        y = alt.Y('y', axis = alt.Axis(grid = False, labels = False, ticks = False)),
        detail = 'edge', **encoded_attrs
    )

    if chart is not None: chart.layer[0] = edge_chart

    return edge_chart



def draw_networkx_nodes(G = None, pos = None,
    chart = None, layer = None,
    nodelist = None, node_size = 300, node_color = 'green', linewidths = 1.0, alpha = 1, cmap = None,
    tooltip = None, **kwargs):
    '''Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    nodelist : list, optional
       Draw only specified nodes

    node_size : scalar or string
       Size of nodes; if an array is specified it must be the
       same length as nodelist.

    node_color : color string, or array of floats
       Node color. Can be a single color format string,
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    node_shape : string
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8'.

    linewidths : float, or array of floats
       Line width for nodes

    alpha : float or array of floats
       The node transparency.  This can be a single alpha value,
       in which case it will be applied to all the nodes of color. Otherwise,
       if it is an array, the elements of alpha will be applied to the colors
       in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap
       Colormap for mapping intensities of nodes

    Returns
    -------
    viz: ``altair.Chart`` object
    '''
    if layer is not None: node_chart = layer
    elif chart is not None:
        df_nodes = chart.layer[1].data
        node_chart = chart.layer[1]
    else:
        df_nodes = to_pandas_nodes(G, pos)
        node_chart = alt.Chart(df_nodes)

    marker_attrs, encoded_attrs = {}, {}


    # ---------- Handle arguments ------------

    # Restrict to a given subset
    if isinstance(nodelist, list):
        df_nodes = node_chart.data = df_nodes.loc[nodelist]
    elif nodelist is not None: raise TypeError('nodelist must be a list or None.')

    # Size
    if isinstance(node_size, str): encoded_attrs['size'] = alt.Size(node_size, legend = None)
    elif isinstance(node_size, int): marker_attrs['size'] = node_size
    else: raise TypeError('node_size must be a string or int.')
    
    # Width of lines
    if isinstance(linewidths, str): encoded_attrs['strokeWidth'] = alt.Size(linewidths, legend = None)
    elif isinstance(linewidths, (float, int)): marker_attrs['strokeWidth'] = linewidths
    else: raise TypeError('linewidths must be a string or int.')

    # Colour
    if not isinstance(node_color, str): raise TypeError('node_color must be a string.')
    if node_color in df_nodes.columns: encoded_attrs['fill'] = node_color
    else: marker_attrs['fill'] = node_color

    # Opacity
    if isinstance(alpha, str): encoded_attrs['opacity'] = alpha
    elif isinstance(alpha, (int, float)): marker_attrs['opacity'] = alpha
    elif alpha is not None: raise TypeError('alpha must be a string or None.')

    # Colour map
    if isinstance(cmap, str):
        encoded_attrs['fill'] = alt.Color(node_color, scale = alt.Scale(scheme = cmap))
    elif cmap is not None: raise TypeError('cmap must be a string (colormap name) or None.')

    # Tooltip
    if tooltip is not None: encoded_attrs['tooltip'] = tooltip


    # ---------- Construct visualization ------------

    node_chart = node_chart.mark_point(**marker_attrs).encode(
        x = alt.X('x', axis = alt.Axis(grid = False, labels = False, ticks = False)),
        y = alt.Y('y', axis = alt.Axis(grid = False, labels = False, ticks = False)),
        **encoded_attrs
    )

    if chart is not None: chart.layer[1] = node_chart

    return node_chart



def draw_networkx_labels(G = None, pos = None,
    chart = None, layer = None,
    nodelist = None, font_size = 15, font_color = 'black', node_label = 'label', **kwargs):
    '''Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    nodelist : list, optional
       Draw only specified nodes

    font_size : scalar or string
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.

    font_color : color string, or array of floats
       Node color. Can be a single color format string,
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    node_label : string
       The name of the node attribute to treat as a label.

    Returns
    -------
    viz: ``altair.Chart`` object
    '''
    if layer is not None: node_chart = layer
    elif chart is not None:
        df_nodes = chart.layer[1].data
        node_chart = chart.layer[1]
    else:
        df_nodes = to_pandas_nodes(G, pos)
        node_chart = alt.Chart(df_nodes)

    marker_attrs, encoded_attrs = {}, {}


    # ---------- Handle arguments ------------

    # Restrict to a given subset
    if isinstance(nodelist, list):
        df_nodes = node_chart.data = df_nodes.loc[nodelist]
    elif nodelist is not None: raise TypeError('nodelist must be a list or None.')

    # Size
    if isinstance(font_size, str): encoded_attrs['size'] = alt.Size(font_size, legend = None)
    elif isinstance(font_size, int): marker_attrs['size'] = font_size
    else: raise TypeError('font_size must be a string or int.')

    # Colour
    if not isinstance(font_color, str): raise TypeError('font_color must be a string.')
    if font_color in df_nodes.columns: encoded_attrs['fill'] = font_color
    else: marker_attrs['fill'] = font_color


    # ---------- Construct visualization ------------

    node_chart = node_chart.mark_text(baseline = 'middle', **marker_attrs).encode(
        x = alt.X('x', axis = alt.Axis(grid = False, labels = False, ticks = False)),
        y = alt.Y('y', axis = alt.Axis(grid = False, labels = False, ticks = False)),
        text = node_label, **encoded_attrs
    )

    if chart is not None: chart.layer[1] = node_chart

    return node_chart



def draw_networkx(G = None, pos = None,
    chart = None,
    nodelist = None, edgelist = None, show_orphans = True, show_self_loops = True,
    node_size = 300, node_color = 'green',
    node_label = None, font_color = 'black', font_size = 15,
    alpha = 1, cmap = None, linewidths = 1.0, width = 1,
    arrow_width = 2, arrow_length = 0.1, arrow_length_is_relative = True,
    edge_color = 'grey', arrow_color = 'black',
    node_tooltip = None, edge_tooltip = None,
    edge_cmap = None, arrow_cmap = None,
    curved_edges = False, control_points = None, interpolate = 'basis'):
    '''Draw the graph G using Altair.

    nodelist : list, optional
       Draw only specified nodes

    edgelist : list, optional
       Draw only specified edges
    
    show_orphans : bool
        Whether to plot nodes with no edges

    show_self_loops : bool
        Whether to plot edges starting and ending on the same node;
        nodes with only self-loops will still be plotted (though edge-less) unless show_orphans is also False
        
    node_size : scalar or array, optional
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.

    node_color : color string, or array of floats,
       Node color. Can be a single color format string,
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    linewidths : float, or array of floats
       Line width for nodes
    
    font_size : scalar or string
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.

    font_color : color string, or array of floats
       Node color. Can be a single color format string,
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    node_label : string
       The name of the node attribute to treat as a label.

    alpha : float, optional
       The node and edge transparency

    cmap : Matplotlib colormap, optional
       Colormap for mapping intensities of nodes

    width : float, optional
       Line width of edges

    arrow_width : float, optional
       The width of arrow portions of edges.

    arrow_length : float, optional
       The proportion of the line to be occupied by the arrow.

    arrow_length_is_relative: bool, optional
        Whether arrow_length should be interpreted as a proportion of its edge length instead of an absolute measure.

    edge_color : colour string or edge attribute name
       In the attribute case, its values need to be colour strings if edge_cmap is None and floats if not None.

    arrow_color : colour string or edge attribute name
       In the attribute case, its values need to be colour strings if edge_cmap is None and floats if not None.

    edge_cmap : Matplotlib colormap, optional
       Colormap for mapping intensities of edges; SILENTLY IGNORED unless edge_color is the name of an edge attribute containing floats

    arrow_cmap : Matplotlib colormap, optional
       Colormap for mapping intensities of edges; SILENTLY IGNORED unless edge_color is the name of an edge attribute containing floats

    curved_edges : bool
        Whether edges should be curved (using control_points and interpolate arguments)
    
    control_points : list[tuple[float, float]]
        The control points to use the interpolation method on; they should be expressed in coordinates relative to their straight edge:
        (proportion of edge length parallel to the edge, proportion of edge length perpendicular (anticlockwise) to the edge).
        E.g. the default value of [(0.5, 0.1)] is a single control point halfway along the edge and 0.1 of its length to the left of it.
    
    interpolate : str
        Interpolation method for curved edges (which are built on control points provided in edge-relative coordinates in curved_edges).
        The default interpolation is a cubic spline (i.e. 'basis').
        Interactive examples of possible values: https://altair-viz.github.io/user_guide/marks/line.html
        Corresponding descriptions: https://d3js.org/d3-shape/curve
        
        Recommendation:
        - 'basis', 'catmull-rom' and 'bundle' (and of course 'linear' and 'monotone') are best since they do not overshoot if control points are close to endpoints
        - the open and closed interpolation varieties are not appropriate for edges
        - the 'natural' cubic and 'cardinal' interpolations tend to overshoot for control points close to endpoints
    '''
    if not (show_self_loops and show_orphans):
        G = deepcopy(G)
        if not show_self_loops: G.remove_edges_from(nx.selfloop_edges(G))
        if not show_orphans: G.remove_nodes_from(list(nx.isolates(G))) # wants a non-generator iterable
    
    if not pos: pos = nx.drawing.layout.spring_layout(G)

    # Edges
    if len(G.edges):
        edges = draw_networkx_edges(G, pos,
            edgelist = edgelist,
            alpha = alpha, width = width, edge_color = edge_color, edge_cmap = edge_cmap,
            curved_edges = curved_edges, control_points = control_points, interpolate = interpolate,
            tooltip = edge_tooltip)

        # Arrows
        if nx.is_directed(G):
            arrows = draw_networkx_arrows(G, pos,
                edgelist = edgelist,
                alpha = alpha, arrow_width = arrow_width, arrow_length = arrow_length, arrow_length_is_relative = arrow_length_is_relative, edge_color = arrow_color, edge_cmap = arrow_cmap,
                curved_edges = curved_edges, control_points = control_points,
                tooltip = edge_tooltip)

    # Nodes
    if len(G.nodes):
        nodes = draw_networkx_nodes(G, pos,
            nodelist = nodelist,
            node_size = node_size, node_color = node_color, alpha = alpha, linewidths = linewidths, cmap = cmap,
            tooltip = node_tooltip)

        # Node labels
        if node_label:
            labels = draw_networkx_labels(G, pos,
                nodelist = nodelist,
                font_size = font_size, font_color = font_color, node_label = node_label)


    # Layer the chart
    viz = []
    if len(G.edges):
        viz.append(edges)
        if nx.is_directed(G): viz.append(arrows)

    if len(G.nodes):
        viz.append(nodes)
        if node_label: viz.append(labels)

    if viz: viz = alt.layer(*viz)
    else: raise ValueError('G does not contain any nodes or edges.')

    return viz


