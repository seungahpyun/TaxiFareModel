# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


import joblib
import numpy as np


from encoders import TimeFeaturesEncoder, DistanceTransformer
from utils import compute_rmse
from data import get_data, clean_data


from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
         ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])], remainder="drop")

        self.pipeline = Pipeline([
        ('preproc', preproc_pipe),
        ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=4))
        ])
        return self

    def crossval(self):
        cv_results = cross_val_score(self.pipeline, self.X, self.y, cv=5, n_jobs=-1)
        return cv_results

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        return joblib.dump(self.pipeline, 'model.joblib')


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri('https://mlflow.lewagon.ai/')
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X = df.drop(columns = "fare_amount")
    y = df["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    train = Trainer(X_train, y_train)
    train.run()
    print(train.evaluate(X_test, y_test))
