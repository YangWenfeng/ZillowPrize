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

def get_feature_outlier_index(pd_series, method):
    """
    :param pd_series:
    :param method:
    :return: index list of outliers
    """
    methods = ['inter_quartile_range', 'small_probability_event']
    assert method in methods

    ret = None
    series = pd_series[pd_series.notnull()]

    if method == methods[0]:
        q1, q3 = get_series_q1q3(pd_series)
        iqr = q3 - q1
        ret = [i for i, v in zip(series.index, series.values) if v < q1-1.5*iqr or v > q3+1.5*iqr]

    elif method == methods[1]:
        llimit, ulimit = get_series_percentile(pd_series, 0.5, 99.5)
        ret = [i for i, v in zip(series.index, series.values) if v > ulimit or v < llimit]

    return ret

def replace_feature_outlier_value(pd_series, indexs, value):
    new_series = pd_series.copy()
    new_series[indexs] = value

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

    outlier_indexs_iqr = get_feature_outlier_index(pd_series, 'inter_quartile_range')
    outlier_indexs_spe = get_feature_outlier_index(pd_series, 'small_probability_event')

    q1, q3 = get_series_q1q3(pd_series)
    llimit, ulimit = get_series_percentile(pd_series)

    print 'Feature Outlier: len(outlier_indexs_iqr) = %d, len(outlier_indexs_spe) = %d' % (
        len(outlier_indexs_iqr), len(outlier_indexs_spe))

    ret['iqr_nan'] = \
        replace_feature_outlier_value(pd_series, outlier_indexs_iqr, np.NAN)

    ret['iqr_median'] = \
        replace_feature_outlier_value(pd_series, outlier_indexs_iqr, pd_series.median())

    ret['iqr_boundary'] = \
        replace_with_iqr_boundary(pd_series, q1, q3)

    ret['spe_nan'] = \
        replace_feature_outlier_value(pd_series, outlier_indexs_spe, np.NAN)

    ret['spe_median'] = \
        replace_feature_outlier_value(pd_series, outlier_indexs_spe, pd_series.median())

    ret['spe_boundary'] = \
        replace_with_boundary(pd_series, llimit, ulimit)

    return ret


if __name__ == '__main__':
    numbers = np.array([np.NAN, 1, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, np.NAN, 22])
    numbers_series = pd.Series(numbers)

    outlier_indexs_iqr = get_feature_outlier_index(numbers_series, 'inter_quartile_range')
    outlier_indexs_spe = get_feature_outlier_index(numbers_series, 'small_probability_event')
    print 'outlier_indexs_iqr', outlier_indexs_iqr
    print 'outlier_indexs_spe', outlier_indexs_spe

    # new_numbers_series = replace_feature_outlier_value(numbers_series, outlier_indexs_spe, np.NaN)
    # print numbers_series, new_numbers_series

    for name, series in generate_feature_replace_outlier(numbers_series).items():
        print name
        print series
