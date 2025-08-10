import pandas as pd
import numpy as np
from config import settings
from logs import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from transformers.dates import DateFeatureExtractor
from transformers.sparse_num import SparseMaster
from typing import Tuple
from utils import clean_column_names


def read_data() -> pd.DataFrame:
    logger.info(f"Reading data from {settings.DATA_PATH}")
    df = pd.read_excel(
        settings.DATA_PATH,
        sheet_name=settings.DATA_SHEET
    )
    return df


def get_fitted_preprocessor() -> ColumnTransformer:

    X_train, _ = get_split()

    cat_transformer = Pipeline(
        steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )

    date_transformer = DateFeatureExtractor(date_columns=settings.DATE_FEATURES)

    num_transfomer = SparseMaster(scaling_method='power', outlier_treatment='remove')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, settings.CAT_FEATURES),
            ('date', date_transformer, settings.DATE_FEATURES),
            ('num', num_transfomer, settings.NUM_FEATURES)
        ],
        remainder='drop'
    )

    return preprocessor.fit(X_train)


def get_split(split:str = "train") -> Tuple[pd.DataFrame, pd.Series]:

    df = read_data()

    cols_renamed = clean_column_names(df)

    X = cols_renamed.drop(columns=[settings.TARGET], axis=1)
    y = cols_renamed[settings.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=settings.TEST_SIZE, 
        random_state=settings.RANDOM_SEED
    )

    return (X_train, y_train) if (split == "train") else (X_test, y_test)



def get_preprocessed_split(split="train") -> Tuple[np.array, np.array]:

    preprocessor = get_fitted_preprocessor()

    X, y = get_split() if split == "train" else get_split(split="test")

    X_preprocessed, y = preprocessor.transform(X), y
    feature_names = preprocessor.get_feature_names_out()
    return X_preprocessed, y, feature_names