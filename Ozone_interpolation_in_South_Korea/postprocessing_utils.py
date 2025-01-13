# general
import pdb

# data science
import numpy as np
import scipy.stats

# own package
import settings


def confidence_interval(data, confidence=0.95):
    """
    https://stackoverflow.com/questions/15033511/compute-a-confidence-
    interval-from-sample-data
    from shasan
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m-h, 5), round(m, 5), round(m+h, 5)


if __name__ == '__main__':
    """
    Test the routine
    """
    for bin_, lower_border in enumerate(settings.bins[:-1]):
        assert get_bin(lower_border) == bin_
