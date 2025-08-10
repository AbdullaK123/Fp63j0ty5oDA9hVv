from pydantic import BaseModel
from typing import List

class Config(BaseModel):
    DATA_PATH: str = 'data/CRM Data.xlsx'
    DATA_SHEET: str = 'Data'
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    MLFLOW_TRACKING_URI: str = "http://localhost:8081"
    TARGET: str = 'Subscribed'
    CAT_FEATURES: List[str] = [
        'Country',
        'Education',
        'Status'
    ]
    DATE_FEATURES: List[str] = [
        'First Contact'
    ]
    NUM_FEATURES: List[str] = [
        'Days from First Contact to Last Contact',
        'Days from First Contact to First Call',
        'Days from First Contact to Signed up for a demo',
        'Days from First Contact to Filled in customer survey',
        'Days from First Contact to Did sign up to the platform',
        'Days from First Contact to Account Manager assigned',
        'Days from First Call to Signed up for a demo',
        'Days from First Call to Filled in customer survey',
        'Days from First Call to Did sign up to the platform',
        'Days from First Call to Account Manager assigned'
    ]


settings = Config()