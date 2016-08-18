import pandas as pd
import numpy as np

from padua.utils import get_protein_id

# Import standard import methods from pandas
from pandas import (read_csv,
        read_excel,
        read_hdf,
        read_sql,
        read_json,
        read_msgpack,
        read_html,
        read_gbq,
        read_stata,
        read_sas,
        read_clipboard,
        read_pickle)


def write_r(df, f, sep=",", index_join="@", columns_join="."):
    """
    Export dataframe in a format easily importable to R

    Index fields are joined with "@" and column fields by "." by default.
    :param df:
    :param f:
    :param index_join:
    :param columns_join:
    :return:
    """

    df = df.copy()
    df.index = ["@".join([str(s) for s in v]) for v in df.index.values]
    df.columns = [".".join([str(s) for s in v]) for v in df.index.values]
    df.to_csv(f, sep=sep)


# Implement write_ functions for standard DataFrame outputs
# for consistency with others avaialable in this IO package

def write_csv(df, f, *args, **kwargs):
    df.to_csv(f, *args, **kwargs)

def write_excel(df, f, *args, **kwargs):
    df.to_excel(f, *args, **kwargs)

def write_hdf(df, f, *args, **kwargs):
    df.to_hdf(f, *args, **kwargs)

def write_sql(df, f, *args, **kwargs):
    df.to_sql(f, *args, **kwargs)

def write_json(df, f, *args, **kwargs):
    df.to_json(f, *args, **kwargs)

def write_msgpack(df, f, *args, **kwargs):
    df.to_msgpack(f, *args, **kwargs)

def write_html(df, f, *args, **kwargs):
    df.to_html(f, *args, **kwargs)

def write_gbq(df, f, *args, **kwargs):
    df.to_gbq(f, *args, **kwargs)

def write_stata(df, f, *args, **kwargs):
    df.to_stata(f, *args, **kwargs)

def write_clipboard(df, f, *args, **kwargs):
    df.to_clipboard(f, *args, **kwargs)

def write_pickle(df, f, *args, **kwargs):
    df.to_pickle(f, *args, **kwargs)