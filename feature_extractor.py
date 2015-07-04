import os
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA, PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


class FeatureExtractor(object):
    def __init__(self):
        
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            # ('rfe', RandomTreesEmbedding(n_estimators=100, sparse_output=False, n_jobs=1)),
            # ('densifier', DenseTransformer()),
            ('pca', PCA(n_components=0.9)),
            # ('anova', SelectKBest(f_regression, k=500)),
        ])

        # self.rfe = 

    def fit(self, X_df, y_array):
        
        X_encoded = X_df

        # uncomment the line below in the submission
        # path = os.path.dirname(__file__)
        X_encoded = X_df
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        X_encoded = X_encoded.drop('weekday', axis=1)
        X_encoded = X_encoded.drop('week', axis=1)
        X_encoded = X_encoded.drop('year', axis=1)
        X_encoded = X_encoded.drop('std_wtd', axis=1)
        X_encoded = X_encoded.drop('WeeksToDeparture', axis=1)        
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)     

        X_array = X_encoded.values

        return self.pipe.fit(X_array)


    def transform(self, X_df):
        
        X_encoded = X_df

        # uncomment the line below in the submission
        # path = os.path.dirname(__file__)
        X_encoded = X_df
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        X_encoded = X_encoded.drop('weekday', axis=1)
        X_encoded = X_encoded.drop('week', axis=1)
        X_encoded = X_encoded.drop('year', axis=1)
        X_encoded = X_encoded.drop('std_wtd', axis=1)
        X_encoded = X_encoded.drop('WeeksToDeparture', axis=1)        
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)     

        X_array = X_encoded.values

        print self.pipe
        return self.pipe.transform(X_array)


# class DataDummizer(object):
#     def __init__(self):
#         pass

#     def fit(self, X_df, y_array):
#         pass

#     def transform(self, X_df):
#         X_encoded = X_df

#         # uncomment the line below in the submission
#         # path = os.path.dirname(__file__)
#         X_encoded = X_df
#         X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
#         X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
#         X_encoded = X_encoded.drop('Departure', axis=1)
#         X_encoded = X_encoded.drop('Arrival', axis=1)
       
#         X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
#         X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
#         X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
#         X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
#         X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
#         X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
#         X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
#         X_encoded = X_encoded.drop('weekday', axis=1)
#         X_encoded = X_encoded.drop('week', axis=1)
#         X_encoded = X_encoded.drop('year', axis=1)
#         X_encoded = X_encoded.drop('std_wtd', axis=1)
#         X_encoded = X_encoded.drop('WeeksToDeparture', axis=1)        
#         X_encoded = X_encoded.drop('DateOfDeparture', axis=1)     

#         X_array = X_encoded.values

#         return X_array