import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            col: col.strip()
            for col in df.columns
        }
    )