import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os


data_path = 'C:/Users/79354/Desktop/data/ACC'
process_path = 'C:/Users/79354/Desktop/data/ACC/process'


os.makedirs(process_path, exist_ok=True)

def filter_primary_samples(df, only_primary=True):
    if only_primary:
        primary_samples = [col for col in df.columns if col[-3:] == "01A"]
        return df[primary_samples]
    else:
        return df


def get_fixed_names(names):
    return names.str.upper().str.replace('-', '.').str[:12]


exp_data = pd.read_csv(os.path.join(data_path, 'exp.tsv'), sep='\t', index_col=0)
methy_data = pd.read_csv(os.path.join(data_path, 'methy.tsv'), sep='\t', index_col=0)
mirna_data = pd.read_csv(os.path.join(data_path, 'mirna.tsv'), sep='\t', index_col=0)


print("Initial data shapes:", exp_data.shape, methy_data.shape, mirna_data.shape)


exp_data = filter_primary_samples(exp_data, only_primary=True)
methy_data = filter_primary_samples(methy_data, only_primary=True)
mirna_data = filter_primary_samples(mirna_data, only_primary=True)


print("After primary sample filtering:", exp_data.shape, methy_data.shape, mirna_data.shape)


exp_data.columns = get_fixed_names(exp_data.columns)
methy_data.columns = get_fixed_names(methy_data.columns)
mirna_data.columns = get_fixed_names(mirna_data.columns)

common_samples = exp_data.columns.intersection(methy_data.columns).intersection(mirna_data.columns)


print("Common samples count:", len(common_samples))


exp_data = exp_data[common_samples]
methy_data = methy_data[common_samples]
mirna_data = mirna_data[common_samples]

print("After selecting common samples:", exp_data.shape, methy_data.shape, mirna_data.shape)


def remove_high_missing_samples(df, threshold=0.2):
    return df.loc[:, df.isnull().mean() < threshold]

exp_data = remove_high_missing_samples(exp_data)
methy_data = remove_high_missing_samples(methy_data)
mirna_data = remove_high_missing_samples(mirna_data)


print("After removing high-missing samples:", exp_data.shape, methy_data.shape, mirna_data.shape)


def remove_high_missing_features(df, threshold=0.2):
    return df.loc[df.isnull().mean(axis=1) < threshold]

exp_data = remove_high_missing_features(exp_data)
methy_data = remove_high_missing_features(methy_data)
mirna_data = remove_high_missing_features(mirna_data)


print("After removing high-missing features:", exp_data.shape, methy_data.shape, mirna_data.shape)




def log_transform(df):
    return np.log1p(df)

exp_data = log_transform(exp_data)
mirna_data = log_transform(mirna_data)


def keep_high_variance_features(df, num_features=2000):
    if len(df) <= num_features:
        return df
    variances = df.var(axis=1)
    top_variances = variances.nlargest(num_features)
    return df.loc[top_variances.index]

exp_data = keep_high_variance_features(exp_data, num_features=2000)
methy_data = keep_high_variance_features(methy_data, num_features=2000)
mirna_data = keep_high_variance_features(mirna_data, num_features=1000)


print("After variance filtering:", exp_data.shape, methy_data.shape, mirna_data.shape)


def knn_impute(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), index=df.index, columns=df.columns)
    return df_imputed

exp_data = knn_impute(exp_data)
methy_data = knn_impute(methy_data)
mirna_data = knn_impute(mirna_data)


def z_score_normalize(df):

    return (df - df.mean(axis=1).values.reshape(-1, 1)) / df.std(axis=1).values.reshape(-1, 1)

exp_data = z_score_normalize(exp_data)
methy_data = z_score_normalize(methy_data)
mirna_data = z_score_normalize(mirna_data)


print("After normalization:", exp_data.shape, methy_data.shape, mirna_data.shape)


exp_data.to_csv(os.path.join(process_path, 'ACC_exp.txt'), sep='\t')
methy_data.to_csv(os.path.join(process_path, 'ACC_methy.txt'), sep='\t')
mirna_data.to_csv(os.path.join(process_path, 'ACC_mirna.txt'), sep='\t')




