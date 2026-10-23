from typing import Any, cast

import altair as alt
import numpy as np
import pandas as pd


def resolve_coordinate_scaling(xs: list[float], ys: list[float], chart_width: float | None, chart_height: float | None) -> tuple[list[float], list[float], float, float]:
    '''Scale given coordinate lists to [0, 1] along each axis and stretch the coordinate corresponding to the larger chart dimension, resolving the chart's dimensions along the way.

    :param xs: The x coordinates to scale (also produce the width if chart_width is None).
    :param ys: The y coordinates to scale (also produce the height if chart_height is None).
    :param chart_width: Width to set the chart to (while maintaining equality between the drawn axes' units); must not be None together with chart_height.
    :param chart_height: Height to set the chart to (while maintaining equality between the drawn axes' units); must not be None together with chart_width.

    :return: The scaled xs and ys, and the resolved chart_width and chart_height
        (so that the graph\'s aspect ratio matches; the shorter axis will have length of approximately 1 + 2 * chart_padding).
    '''
    if chart_width is None and chart_height is None:
        raise ValueError('chart_width and chart_height cannot both be None; if one is None then the other is determined by the graph\'s own aspect ratio.')

    old_xs, old_ys = xs, ys
    x_range, y_range = max(xs) - min(xs), max(ys) - min(ys)
    if chart_width is None: chart_width = cast(float, chart_height) * x_range / y_range # the cast is for mypy clarity
    if chart_height is None: chart_height = chart_width * y_range / x_range

    # Scale x and y to [0,1] (so that loop_radius values are consistent across charts)
    xs = [0. if x_range == 0 else (x - min(old_xs)) / x_range for x in xs]
    ys = [0. if y_range == 0 else (y - min(old_ys)) / y_range for y in ys]
    if chart_width != chart_height: # Stretch the coordinate corresponding to the larger chart size so that aspect ratios match (and therefore x and y axes' units are equal)
        xs = [x * chart_width/chart_height for x in xs] if chart_width >= chart_height else xs
        ys = ys if chart_width >= chart_height else [y * chart_height/chart_width for y in ys]

    return xs, ys, chart_width, chart_height


def assemble_padded_layers(layers: list[alt.Chart], chart_width: float, chart_height: float, chart_padding: float) -> alt.LayerChart:
    '''Assemble the given layers into a single chart whose axes match the given chart\'s aspect ratio after padding.

    :param layers: The layers to assemble (non-empty; their dataframes must have 'x' and 'y' columns).
    :param chart_width: Width to set the chart to (while maintaining equality between the drawn axes' units).
    :param chart_height: Height to set the chart to (while maintaining equality between the drawn axes' units).
    :param chart_padding: Proportion of the larger of the two axes to use as padding between graph and chart borders
        (followed by a slight padding increase due to final aspect ratio adjustments).

    :return: A single-chart layer of the given layers, with both axes\' domains covering the drawn elements with the given proportion of asymmetric-but-uniform padding.
    '''
    # Extract the coordinate ranges from the layers which are present (since drawn elements, e.g. self-loops or large node sizes, might not match node coordinate ranges)
    min_xs, max_xs, min_ys, max_ys = zip(*((min(l.data['x']), max(l.data['x']), min(l.data['y']), max(l.data['y'])) for l in layers), strict = True)
    x_range = (max_x := max(max_xs)) - (min_x := min(min_xs))
    y_range = (max_y := max(max_ys)) - (min_y := min(min_ys))

    # Compute new coordinate ranges with uniform padding AND s.t. the aspect ratio matches the required one
    #   It is cleaner to work in terms of longer and shorter sides rather than real x and y; reassign at the end
    long_axis, short_axis, long_side, short_side = (x_range, y_range, chart_width, chart_height) if chart_width >= chart_height else (y_range, x_range, chart_height, chart_width)
    long_padding = short_padding = chart_padding * max(x_range, y_range) # use the larger of the two as uniform padding

    # Impose the chart's aspect ratio onto the axes by increasing the length of one of the two (i.e. avoid going below the stated padding; only above)
    if (raw_long := long_axis + 2 * long_padding) / (raw_short := short_axis + 2 * short_padding) >= long_side / short_side:
        short_padding = ((raw_long * short_side / long_side) - short_axis) / 2 # i.e. (correct-aspect-ratio raw_short - short_axis) / 2
    else: long_padding = ((raw_short * long_side / short_side) - long_axis) / 2 # i.e. (correct-aspect-ratio raw_long - long_axis) / 2

    x_padding, y_padding = (long_padding, short_padding) if chart_width >= chart_height else (short_padding, long_padding)

    return alt.layer(*layers).encode(
        alt.X().scale(domain = (min_x - x_padding, max_x + x_padding)),
        alt.Y().scale(domain = (min_y - y_padding, max_y + y_padding))
    ).properties(width = chart_width, height = chart_height)


def copy_size_and_axes(source_chart: alt.Chart, target_chart: alt.Chart) -> alt.Chart:
    '''Set the height & width, as well as the x and y axes' ranges of the target_chart to the same values as the source_chart (ASSUMING they are all defined, as is the case for outputs of draw_networkx).
    This is useful to preserve the drawn aspect ratio when reassembling or concatenating layers with other charts.
    '''
    return target_chart.encode(
        alt.X().scale(domain = source_chart.encoding.x['scale']['domain']),
        alt.Y().scale(domain = source_chart.encoding.y['scale']['domain'])
    ).properties(width = source_chart.width, height = source_chart.height)



def is_arraylike(obj: Any) -> bool:
    '''Return True if array-like (accepts lists, numpy.ndarray, pandas.Series, pandas.DataFrame).
    '''
    return isinstance(obj, (list, np.ndarray, pd.Series, pd.DataFrame))



def despine(chart: alt.Chart) -> alt.Chart:
    '''Despine an altair chart (i.e. remove ticks, grid, domain and labels).
    '''
    return chart.configure_axis(ticks = False, grid = False, domain = False, labels = False)



def validate_snapshot_schemas(dfs: list[pd.DataFrame], frames: list[Any], entity: str) -> None:
    '''Ensure that all snapshots' DataFrames of an entity type carry the same attribute columns of the same dtypes (excluding the stitching columns).

    :param dfs: One DataFrame per snapshot for the entity type (snapshots without rows of this entity type are skipped).
    :param frames: The DataFrames' frame values in playback order (used to report the offending snapshots).
    :param entity: Name of the entity type (used to report, e.g. 'node').
    '''
    if not (snapshots := [(df, frame) for df, frame in zip(dfs, frames, strict = True) if len(df)]): return
    base, base_frame = snapshots[0]
    attribute_columns = [c for c in base.columns if c not in ('frame_index', 'frame')]
    problems = []
    for df, frame in snapshots[1:]:
        missing = [c for c in attribute_columns if c not in df.columns]
        extra = [c for c in df.columns if c not in attribute_columns and c not in ('frame_index', 'frame')]
        mismatched = {c: (base.dtypes[c], df.dtypes[c]) for c in attribute_columns if c in df.columns and df.dtypes[c] != base.dtypes[c]}
        if missing or extra or mismatched: problems.append(f'vs frame {base_frame!r} (the first snapshot in playback order): the {entity} attributes are missing {missing}, unexpected {extra}, mismatched dtypes {mismatched}; offending frame: {frame!r}')
    if problems: raise ValueError('the snapshots must carry identical attribute columns AND dtypes per entity type for stitching; align them across the graphs (add/drop/fill the missing attributes) and stitch again:\n' + '\n'.join(problems))
