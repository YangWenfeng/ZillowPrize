import numpy as np
import pandas as pd
from collections import OrderedDict

__author__ = 'yangwenfeng'


def get_series_percentile(pd_series, lpercentile=0.5, upercentile=99.5):
    series = pd_series[pd_series.notnull()]

    llimit = np.percentile(series.values, lpercentile)
    ulimit = np.percentile(series.values, upercentile)

    return llimit, ulimit

def get_series_q1q3(pd_series):
    return get_series_percentile(pd_series, 25, 75)

def replace_with_iqr_value(pd_series, q1, q3, value):
    iqr = q3 - q1
    llimit = q1 - 1.5 * iqr
    ulimit = q3 + 1.5 * iqr

    return replace_with_value(pd_series, llimit, ulimit, value)

def replace_with_value(pd_series, llimit, ulimit, value):
    new_series = pd_series.copy()

    new_series.loc[new_series > ulimit] = value
    new_series.loc[new_series < llimit] = value

    return new_series

def replace_with_iqr_boundary(pd_series, q1, q3):
    iqr = q3 - q1
    llimit = q1 - 1.5 * iqr
    ulimit = q3 + 1.5 * iqr

    return replace_with_boundary(pd_series, llimit, ulimit)

def replace_with_boundary(pd_series, llimit, ulimit):
    new_series = pd_series.copy()

    new_series.loc[new_series > ulimit] = ulimit
    new_series.loc[new_series < llimit] = llimit

    return new_series

def generate_feature_replace_outlier(pd_series):
    ret = OrderedDict()

    q1, q3 = get_series_q1q3(pd_series)
    llimit, ulimit = get_series_percentile(pd_series)

    ret['iqr_nan']      = replace_with_iqr_value(pd_series, q1, q3, np.NAN)
    ret['iqr_median']   = replace_with_iqr_value(pd_series, q1, q3, pd_series.median())
    ret['iqr_mean']     = replace_with_iqr_value(pd_series, q1, q3, pd_series.mean())
    ret['iqr_boundary'] = replace_with_iqr_boundary(pd_series, q1, q3)

    ret['spe_nan']      = replace_with_value(pd_series, llimit, ulimit, np.NAN)
    ret['spe_median']   = replace_with_value(pd_series, llimit, ulimit, pd_series.median())
    ret['spe_mean']     = replace_with_value(pd_series, llimit, ulimit, pd_series.mean())
    ret['spe_boundary'] = replace_with_boundary(pd_series, llimit, ulimit)

    return ret


if __name__ == '__main__':
    numbers = np.array([np.NAN, 1, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, np.NAN, 22])
    numbers_series = pd.Series(numbers)

    for name, series in generate_feature_replace_outlier(numbers_series).items():
        print name
        print series
