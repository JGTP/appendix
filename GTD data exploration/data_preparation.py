import os
from datetime import datetime

import numpy as np
import pandas as pd
import prince
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder

import feature_selection as fs


def prepare(raw_data_path, test=False, clustering=False, n_components=2):
    print("Preparing data...")
    df = pd.read_excel(raw_data_path, index_col='eventid')
    df = exclude_columns(df)
    df = clean_data(df)
    if clustering:
        df = standardise(df)
        # print(df.select_dtypes(np.number).columns.tolist())
        cols = fs.date_cols + fs.geo_cols + \
            fs.coordinates + fs.text_cols
        columns = [c for c in cols if c in df.columns]
        df = df.drop(columns, axis=1)
        df = dimension_reduction(df, n_components)
    else:
        cols = fs.date_cols + fs.geo_cols + fs.coordinates
        columns = [c for c in cols if c in df.columns]
        df = df.drop(columns, axis=1)
        df = encode(df, hot=True)
    print('Number of features after data preparation: ' + str(len(df.columns)))
    return df


def standardise(df):
    print('Standardising numerical data...')
    cols = fs.numerical_cols
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
    return df


def dimension_reduction(X, n_components):
    print("Applying FAMD...")
    X = prince.FAMD(n_components=n_components, n_iter=3, copy=True,
                    check_input=True, engine='auto', random_state=42).fit_transform(X)
    # ax = famd.plot_row_coordinates(X,ax=None,figsize=(6, 6),x_component=0,
    #     y_component=1,ellipse_outline=True,ellipse_fill=True,show_points=False)
    # ax.get_figure().savefig('famd_row_coordinates.svg')
    print("Number of features after dimension reduction: " + str(len(X.columns)))
    return X


def exclude_columns(df):
    print('Filtering dataframe...')
    cols = fs.exclude_columns + fs.exclude_categorical_number_columns + \
        fs.excluded_categorical_string_cols
    df.drop(cols, axis=1, inplace=True)
    return df


def clean_data(X):
    print('Cleaning problematic features...')
    X = handle_numerical_unknowns(X)
    X = handle_categorical_variables(X)
    X = handle_numerical_variables(X)
    X[fs.coordinates] = X[fs.coordinates].apply(handle_coordinates)
    return X


def handle_coordinates(row):
    row = pd.to_numeric(row, errors='coerce')
    row = row.apply(lambda long: long / 1000 if abs(long) > 1000 else long)
    row = row.apply(lambda long: long / 1000 if abs(long) > 1000 else long)
    return row


def handle_numerical_unknowns(X):
    cols = fs.minus_9_is_unknown + fs.minus_99_is_unknown
    X[cols] = X[cols].apply(replace_unknown)
    return X


def handle_numerical_variables(X):
    numerical_cols = fs.numerical_cols
    X[numerical_cols] = X[numerical_cols].fillna(
        0).apply(pd.to_numeric, downcast='integer')
    return X


def handle_categorical_variables(X):
    categorical_string_cols = fs.categorical_string_cols
    X[categorical_string_cols] = X[categorical_string_cols].fillna('Unknown')
    categorical_number_columns = fs.categorical_number_columns
    X[categorical_number_columns] = X[categorical_number_columns].fillna(
        0).apply(pd.to_numeric, downcast='integer')
    X[categorical_string_cols + categorical_number_columns] = X[categorical_string_cols +
                                                                categorical_number_columns].astype('category')
    return X


def construct_custom_features(data):
    data['weekday'] = data.apply(get_weekday, axis=1)
    return data


def get_weekday(row):
    # Sometimes the (precise) date is unknown.
    if row['iday'] == 0:
        return -1
    else:
        date = str(row['iyear']) + str(row['imonth']) + str(row['iday'])
        date = datetime.strptime(date, "%Y%m%d")
        return date.weekday()


def replace_unknown(row):
    row = row.replace({-99.0: None, -9: None})
    return row


def encode(X, hot):
    print('Encoding categorical features...')
    categorical_columns = fs.categorical_string_cols + fs.categorical_number_columns
    for col in categorical_columns:
        print('Encoding "%s" (%i/%i)' %
              (col, categorical_columns.index(col) + 1, len(categorical_columns)))
        if hot:
            col = [col]
            X = pd.get_dummies(X, columns=col, drop_first=True, sparse=False)
        else:
            X[col] = encode_labels(X[col])
    print('Encoding completed.')
    return X


def encode_labels(series):
    enc = LabelEncoder()
    series = enc.fit_transform(series)
    return series
