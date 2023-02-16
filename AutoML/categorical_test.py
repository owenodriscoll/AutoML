#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:51:26 2023

@author: owen
"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from AutoML.AutoML.scalers_transformers import PcaChooser, PolyChooser, SplineChooser, ScalerChooser, \
    TransformerChooser






from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

numeric_features = ["age", "fare"]
ordinal_columns = ['ten']
categorical_columns = ['nine']


poly = PolyChooser({"interaction_only" : True}).fit()
scaler = StandardScaler()
pca = PcaChooser(pca_value=0.95).fit()



from sklearn.linear_model import LassoLars

one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
ordinal_encoder = OrdinalEncoder()


class CategoricalChooser:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ordinal_encoder = OrdinalEncoder()
    
    
    def __init__(self, ordinal_columns, categorical_columns):
        self.ordinal_columns = ordinal_columns
        self.categorical_columns = categorical_columns


    def fit(self):
        
        if type(ordinal_columns) == type(categorical_columns) == list:
            transformers = [
                ("ordinal", ordinal_encoder, ordinal_columns),
                ("nominal", one_hot_encoder, categorical_columns),
            ]
        elif  type(ordinal_columns) != type(categorical_columns) == list:
            transformers = [
                ("nominal", one_hot_encoder, categorical_columns),
            ]
        elif  type(categorical_columns) != type(ordinal_columns) == list:
            transformers = [
               ("ordinal", ordinal_encoder, ordinal_columns),
            ]
        else:
            transformers = None
        
        self.func_fitted = ColumnTransformer(
                transformers=transformers,
                remainder="passthrough", verbose_feature_names_out=False,
            )
        
        return self.func_fitted
        

ColumnTransformer(
        transformers=None,
        remainder="passthrough", verbose_feature_names_out=False,
    )



pipeline = Pipeline([
    # ("ordinal", ordinal),
    ("cat", categorical),
    ('poly', poly), 
    # ('spline', spline), 
    ('scaler', scaler), ('pca', pca), 
    ("reg", LassoLars()),
])


pipeline.fit_transform(test.X_train[['nine', 'ten']], test.y_train).shape
pipeline.fit(test.X_train[['nine', 'ten']], test.y_train)
pipeline.score(test.X_test, test.y_test)


