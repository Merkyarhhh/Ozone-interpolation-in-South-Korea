"""
Utils for spatial patterns
"""

# general
import pdb

# data science
import numpy as np
import pandas as pd


def query_db(query):
    """
    Returns a pandas dataframe with the queried data
    Note: imports do not make sense everywhere, so they happen inside
    the function!
    """
    # database access imports
    from private_database_credentials import db_user, db_password, \
                                             db_host, db_port, db_name

    # connector imports
    import psycopg2
    import pandas.io.sql as sqlio
    try:
        connection = psycopg2.connect(user=db_user,
                                      password=db_password,
                                      host=db_host,
                                      port=db_port,
                                      database=db_name)
        data = sqlio.read_sql_query(query, connection)
        return data
    except Exception as exc:
        print('Database query failed')
        print(exc)
        exit()


def print_statistics(name, data):
    """
    Basic statistics for data sanity checks
    """
    # read in data
    data = np.array(data)

    # calculate statistics
    if data.dtype in ['int64', 'float64']:
        miss_ = np.count_nonzero(np.isnan(data)) / data.size * 100.
        min_ = np.nanmin(data)
        max_ = np.nanmax(data)
        mean_ = np.nanmean(data)
        median_ = np.nanmedian(data)

        s = ''
        s += f'missing: {miss_:.2f} %  '
        s += f'min: {min_:.2f}  max: {max_:.2f}  '
        s += f'mean: {mean_:.2f}  median: {median_:.2f}'

    else:
        miss_ = np.count_nonzero(data=='') / data.size * 100
        unique_ = len(np.unique(data))

        s = ''
        s += f'missing: {miss_:.2f} %  '
        s += f'unique: {unique_}  '

    # print statistics
    print(f'Basic statistics for {name}:')
    print(s)
