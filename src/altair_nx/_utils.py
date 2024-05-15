import numpy as np
import pandas as pd


def is_arraylike(obj):
    '''Return True if array-like (accepts lists, numpy.ndarray, pandas.Series, pandas.DataFrame).
    '''
    return isinstance(obj, (list, np.ndarray, pd.Series, pd.DataFrame))



def despine(chart):
    '''Despine altair chart.
    '''
    return chart.configure_axis(ticks = False, grid = False, domain = False, labels = False)


