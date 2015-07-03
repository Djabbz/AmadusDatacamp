import os
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

class FeatureExtractor(object):
    def __init__(self):
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('kpca', KernelPCA(n_components=0.7)),
            # ('anova', SelectKBest(f_regression, k=1500)),
        ])

        self.kernel_pca = KernelPCA(n_components=0.7)

    def fit(self, X_df, y_array):
        X_encoded = X_df
        
        #uncomment the line below in the submission
        #path = os.path.dirname(__file__)
        X_encoded = X_df
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
    
     #   data_holidays = pd.read_csv(os.path.join(path, "data_holidays.csv"))
     #   X_holidays = data_holidays[['DateOfDeparture','Xmas','Xmas-1','NYD','NYD-1','Ind','Thg','Thg+1']]
     #   X_encoded = X_encoded.set_index(['DateOfDeparture'])
     #   X_holidays = X_holidays.set_index(['DateOfDeparture'])
     #   X_encoded = X_encoded.join(X_holidays).reset_index()        
        
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
        self.pipe.fit(X_array)

    def transform(self, X_df):
        X_encoded = X_df
        
        #uncomment the line below in the submission
        #path = os.path.dirname(__file__)
        X_encoded = X_df
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
    
     #   data_holidays = pd.read_csv(os.path.join(path, "data_holidays.csv"))
     #   X_holidays = data_holidays[['DateOfDeparture','Xmas','Xmas-1','NYD','NYD-1','Ind','Thg','Thg+1']]
     #   X_encoded = X_encoded.set_index(['DateOfDeparture'])
     #   X_holidays = X_holidays.set_index(['DateOfDeparture'])
     #   X_encoded = X_encoded.join(X_holidays).reset_index()        
        
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
        self.pipe.transform (X_array)

        return X_array