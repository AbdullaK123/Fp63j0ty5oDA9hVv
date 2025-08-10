import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from etl import get_preprocessed_split
from config import settings
from logs import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from shap.plots import bar, beeswarm

def train_model():

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    mlflow.set_experiment("Subscription Classifier")

    with mlflow.start_run() as run:

        X_train, y_train, feature_names = get_preprocessed_split(split="train")
        X_test, y_test, _ = get_preprocessed_split(split="test")

        model = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 50),    
            cv=5,                             
            class_weight='balanced',          
            scoring='f1'                      
        )

        model.fit(X_train, y_train)
        

        # log importances and shap values

        # importances
        feature_importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.coef_[0],
            'abs_importance': abs(model.coef_[0])
        }).sort_values('abs_importance', ascending=False)

        feature_importances_json = feature_importances_df.to_json(orient='records', indent=2)

        with open('feature_importances.json', 'w') as f:
            f.write(feature_importances_json)

        mlflow.log_artifact('feature_importances.json')

        #shap values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap_values = shap.Explanation(
            values=np.asarray(shap_values.values, dtype=float),
            base_values=np.asarray(shap_values.base_values, dtype=float),
            data=np.asarray(X_test, dtype=float),
            feature_names=feature_names
        )

        beeswarm(shap_values, max_display=20)
        plt.savefig('shap_values_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact('shap_values_beeswarm.png')

        bar(shap_values, max_display=20)
        plt.savefig('shap_values_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact('shap_values_bar.png')
        
        np.save("shap_values.npy", shap_values.values)  # save as .npy

        # Then log with mlflow
        mlflow.log_artifact("shap_values.npy")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('roc_auc', roc_auc)
        mlflow.log_dict(report, 'classification_report.json')
        mlflow.sklearn.log_model(model, signature=signature, artifact_path='model')

        logger.info(f"Finished run: {run.info.run_id}")
        
