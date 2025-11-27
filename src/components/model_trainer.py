import json
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object, save_object, read_yaml_file
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel
from src.constants import MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self._model_config = read_yaml_file(file_path=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_model_object_and_report(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a LogisticRegression model after Vectorization.

        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training LogisticRegression model after Vectorization.")

            # Remove NaN rows
            train = train.dropna()
            test = test.dropna()

            # Splitting the train and test data into features and target variables
            x_train, x_test = train.iloc[:, 0].astype(str), test.iloc[:, 0].astype(str)
            y_train, y_test = train.iloc[:, -1], test.iloc[:, -1]
            logging.info("train-test split done.")

            vec_params = dict(self._model_config["model"]["vectorizer"]["params"])
            clf_params = dict(self._model_config["model"]["classifier"]["params"])

            # Convert nrgam_range list → tuple
            if "ngram_range" in vec_params:
                vec_params["ngram_range"] = tuple(vec_params["ngram_range"])

            # Initialize the Pipeline with specified parameters
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(**vec_params)),
                ("clf", LogisticRegression(**clf_params))
            ])

            # Fit the model
            logging.info("Model training going on...")
            pipeline.fit(x_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = pipeline.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(accuracy=accuracy,f1_score=f1, precision_score=precision, recall_score=recall)
            return pipeline, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_data = self.read_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_data = self.read_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_data, test=test_data)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            x_train_data = train_data.iloc[:, 0].astype(str)
            y_train_data = train_data.iloc[:, -1]
            train_accuracy = accuracy_score(y_train_data, trained_model.predict(x_train_data))

            if train_accuracy < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performance is better than previous one.")
            final_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, final_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            # Save the model score report
            model_score_report = {
                "Accuracy": metric_artifact.accuracy,
                "F1_Score": metric_artifact.f1_score,
                "Precision_Score": metric_artifact.precision_score,
                "Recall_Score": metric_artifact.recall_score
            }

            with open(self.model_trainer_config.trained_model_score_path, "w") as report_file:
                json.dump(model_score_report, report_file, indent=4)
            logging.info("Model score report saved.")

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e