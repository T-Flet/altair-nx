import numpy as np
import pandas as pd
import altair as alt


def copy_size_and_axes(source_chart: alt.Chart, target_chart: alt.Chart):
    '''Set the height & width, as well as the x and y axes' ranges of the target_chart to the same values as the source_chart (ASSUMING they are all defined, as is the case for outputs of draw_networkx).
    This is useful to preserve the drawn aspect ratio when reassembling or concatenating layers with other charts.
    '''
    return target_chart.encode(
        alt.X().scale(domain = source_chart.encoding.x['scale']['domain']),
        alt.Y().scale(domain = source_chart.encoding.y['scale']['domain'])
    ).properties(width = source_chart.width, height = source_chart.height)



def is_arraylike(obj):
    '''Return True if array-like (accepts lists, numpy.ndarray, pandas.Series, pandas.DataFrame).
    '''
    return isinstance(obj, (list, np.ndarray, pd.Series, pd.DataFrame))



def despine(chart: alt.Chart):
    '''Despine an altair chart (i.e. remove ticks, grid, domain and labels).
    '''
    return chart.configure_axis(ticks = False, grid = False, domain = False, labels = False)


