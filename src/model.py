import argparse

import mlflow
from sklearn.linear_model import ElasticNet

from src.config import CFG
from src.data_loader import load_data
from src.utils import split_dataset, eval_function


def main(alpha, l1_ratio):
    df = load_data()
    X_train, X_test, y_train, y_test = split_dataset(df)

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
        mlflow.sklearn.log_model(model, "trained_model")


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--alpha", "-a", type=float, default=CFG.alpha_param)
    args.add_argument("--l1_ratio", "-l1", type=float, default=CFG.l1_ratio)
    parsed_args = args.parse_args()

    main(parsed_args.alpha, parsed_args.l1_ratio)
