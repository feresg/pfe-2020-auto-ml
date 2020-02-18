from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import squareform, pdist
import pandas as pd
import numpy as np
from metafeatures import Metafeatures

# this file is only for testing...

# testing on iris dataset
data = load_iris()

df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                  columns=data['feature_names'] + ['target'])


mf = Metafeatures(df, 'target', task='classification')
all_mfs = mf.compute()
print(all_mfs)
print(len(all_mfs))

# testing on boston dataset
boston = load_boston()

df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                  columns=data['feature_names'] + ['target'])


mf = Metafeatures(df, 'target', task='regression')
all_mfs = mf.compute()
print(all_mfs)
print(len(all_mfs))

df1, df2 = train_test_split(df, test_size=0.5)

mf1 = Metafeatures(df1, 'target', task='regression')
all_mfs1 = mf1.compute()

mf2 = Metafeatures(df2, 'target', task='regression')
all_mfs2 = mf2.compute()

mf_dataframe = pd.DataFrame(all_mfs1, index=[0])
mf_dataframe = mf_dataframe.append(all_mfs2, ignore_index=True)
print(mf_dataframe)


distances_df = pd.DataFrame(squareform(
    pdist(mf_dataframe, 'minkowski', p=1.)))

print(distances_df)
