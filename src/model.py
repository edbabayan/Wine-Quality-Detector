import argparse

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.linear_model import ElasticNet

from config import CFG
from data_loader import load_data
from utils import split_dataset, eval_function


def main(alpha, l1_ratio):
    df = load_data()
    X_train, X_test, y_train, y_test = split_dataset(df)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("ML-Model-1")
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=6)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_function(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2-score", r2)

        # Create an input example and infer signature
        input_example = X_train[:5]  # Provide a small slice of the training data
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model with input example and signature
        mlflow.sklearn.log_model(
            model,
            "trained_model",
            signature=signature,
            input_example=input_example
        )


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--alpha", "-a", type=float, default=CFG.alpha_param)
    args.add_argument("--l1_ratio", "-l1", type=float, default=CFG.l1_ratio)
    parsed_args = args.parse_args()

    main(parsed_args.alpha, parsed_args.l1_ratio)
