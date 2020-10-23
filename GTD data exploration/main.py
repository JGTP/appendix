import os
import warnings

import pandas as pd
import tqdm
from pylab import rcParams

import circumstances
import clustering
import data_preparation
import feature_selection
import mo_classification

warnings.simplefilter(action='ignore', category=FutureWarning)


def label_data(rawDataPath, outputSampling=False, test=False):

    if os.path.isfile(rawDataPath) == False:
        if test == False:
            prepare_raw_data(
                "../../OWL/MOs/GTD_data/globalterrorismdb_0919dist.xlsx", False)
        else:
            prepare_raw_data(
                "../../OWL/MOs/GTD_data/test.xlsx", True)
    if outputSampling:
        outputPath = "../../OWL/MOs/GTD_data/labelledDataSample.xlsx"
    else:
        outputPath = "../../OWL/MOs/GTD_data/labelledData.xlsx"
    try:
        os.remove(outputPath)
        print('Existing output file removed.')
    except FileNotFoundError:
        print('No existing output file found.')
    print('Reading raw data input...')
    df = pd.read_excel(rawDataPath, header=0, index_col='eventid')
    categorise_MO(df, outputPath)


def prepare_raw_data(dataPath, test):
    print('Preparing raw data...')
    df = pd.read_excel(dataPath, header=0, index_col='eventid')
    if test == False:
        df = df[df['iyear'] > 1997]
    print('Constructing circumstances...')
    tqdm.tqdm.pandas()
    df['circumstance_time'] = df.progress_apply(
        circumstances.construct_time, axis=1)
    df['circumstance_nvictims'] = df.progress_apply(
        circumstances.construct_nvictims, axis=1)
    df['circumstance_location'] = df.progress_apply(
        circumstances.construct_location, axis=1)
    df['circumstance_target'] = df.progress_apply(
        circumstances.construct_target, axis=1)
    df['targtype1_txt'] = df['targtype1_txt'].str.replace(",", ".")
    df['targtype1_txt'] = df['targtype1_txt'].str.replace("(", "_")
    df['targtype1_txt'] = df['targtype1_txt'].str.replace(")", "_")
    df['targtype1_txt'] = df['targtype1_txt'].str.replace(":", "--")
    df['targsubtype1_txt'] = df['targsubtype1_txt'].str.replace(",", ".")
    df['targsubtype1_txt'] = df['targsubtype1_txt'].str.replace("(", "_")
    df['targsubtype1_txt'] = df['targsubtype1_txt'].str.replace(")", "_")
    df['targsubtype1_txt'] = df['targsubtype1_txt'].str.replace(":", "--")
    df['gname'] = df['gname'].str.replace(",", ".")
    df['gname'] = df['gname'].str.replace("(", "_")
    df['gname'] = df['gname'].str.replace(")", "_")
    df['gname'] = df['gname'].str.replace(":", "--")
    df['weaptype1_txt'] = df['weaptype1_txt'].str.replace(",", ".")
    df['weaptype1_txt'] = df['weaptype1_txt'].str.replace("(", "_")
    df['weaptype1_txt'] = df['weaptype1_txt'].str.replace(")", "_")
    df['weaptype1_txt'] = df['weaptype1_txt'].str.replace(":", "--")
    df['weapsubtype1_txt'] = df['weapsubtype1_txt'].str.replace(",", ".")
    df['weapsubtype1_txt'] = df['weapsubtype1_txt'].str.replace("(", "_")
    df['weapsubtype1_txt'] = df['weapsubtype1_txt'].str.replace(")", "_")
    df['weapsubtype1_txt'] = df['weapsubtype1_txt'].str.replace(":", "--")
    print("Writing to raw data file...")
    df.to_excel(rawDataPath, index=True)


def categorise_MO(df, outputPath, outputSampling=False):
    print('Categorising MOs...')
    tqdm.tqdm.pandas()
    df['label'] = df.progress_apply(categorise, axis=1)
    df.dropna(subset=['label'], inplace=True)
    df['index'] = '\'' + df.index.astype(str) + '\''
    print('Writing labelled data output...')
    if outputSampling:
        print('Limiting output...')
        df = df.head(100)
    df.to_excel(outputPath, header=True, index=True)
    print('Saved to excel')


def categorise(row):
    dftax = pd.read_excel("../../OWL/MOs/GTD_data/taxonomy.xlsx")
    attacktype1 = row['attacktype1_txt']
    suicide = row['suicide']
    weapsubtype1 = row['weapsubtype1_txt']
    claimed = row['claimed']
    targtype1 = row['targtype1_txt']
    label = dftax[(dftax.attacktype1 == attacktype1) & (dftax.suicide == suicide) & (dftax.weapsubtype1 ==
                                                                                     weapsubtype1) & (dftax.claimed == claimed) & (dftax.targtype1 == targtype1)]['Fifth level'].values
    if len(label) > 0:
        # print(label[0])
        return label[0]


def classify(raw_data_path, feature, groups=[], test=False):
    df = data_preparation.prepare(raw_data_path, test)
    rcParams['figure.figsize'] = 20, 25
    df = df[df[feature] != 'Unknown']
    mo_classification.classify_group(df, feature, groups)


# def run_clustering(raw_data_path, n_components=2, test=False):
#     rcParams['figure.figsize'] = 5, 5
#     df = data_preparation.prepare(
#         raw_data_path, test, clustering=True, n_components=n_components)
#     cols = feature_selection.date_cols + feature_selection.geo_cols + \
#         feature_selection.coordinates + feature_selection.text_cols
#     columns = [c for c in cols if c in df.columns]
#     df = df.drop(columns, axis=1)
#     # df['cluster'] = clustering.k_prototypes(df, n_clusters)
#     # print(df.isnull().any())
#     rcParams['figure.figsize'] = 20, 25
#     df['cluster'] = clustering.hierarchical_clustering(df)
#     # print(df.head())x


# Run with Jupyter notebook
rawDataPath = "../../OWL/MOs/GTD_data/rawData.xlsx"
label_data(rawDataPath, test=False)
# group1 = 'Islamic State of Iraq and the Levant (ISIL)'
# group2 = "Taliban"
# classify(rawDataPath, 'gname', groups=[group1, group2])
