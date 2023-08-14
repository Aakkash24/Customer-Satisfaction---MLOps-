import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE,RMSE,R2
from sklearn.base import RegressorMixin
from typing import Tuple
import mlflow
from typing_extensions import Annotated
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin,
                   X_test :pd.DataFrame,
                   Y_test:pd.Series) -> Tuple[
                       Annotated[float,"r2_score"],
                       Annotated[float,"rmse"],
                   ]:
    """
    Evaluate the model on the ingestion data
    Args:
        df:ingestion data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(prediction, Y_test)
        mlflow.log_metric("mse",mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(prediction, Y_test)
        mlflow.log_metric("r2",r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(prediction,Y_test)
        mlflow.log_metric("rmse",rmse)

        return r2,rmse
    except Exception as e:
        logging.error("Error in evaluating the mode : {}".format(e))
        raise e
