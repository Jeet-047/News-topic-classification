from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object, is_model_present, save_object
import sys
import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float
    model: object


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self):
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available on the model directory
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if is_model_present(self.model_eval_config.final_model_file_path):
                return load_object(file_path=self.model_eval_config.final_model_file_path)
            else:
                return None
        except Exception as e:
            raise  MyException(e,sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Loading the transformed test data for prediction...")
            test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_file_path)
            test_df_cleaned = test_df.dropna()  # remove the empty values
            x, y = test_df_cleaned.drop(TARGET_COLUMN, axis=1), test_df_cleaned[TARGET_COLUMN]

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model, average="weighted")
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            else:
                best_model = trained_model
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted= tmp_best_model_score > trained_model_f1_score,
                                           difference= tmp_best_model_score - trained_model_f1_score,
                                           model=best_model
                                           )
            
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            
            logging.info("Model evaluation completed. Now, save the best model.")
            if evaluate_model_response.is_model_accepted:
                os.makedirs(os.path.dirname(self.model_eval_config.final_model_file_path), exist_ok=True)
                save_object(file_path=self.model_eval_config.final_model_file_path, obj=evaluate_model_response.model)

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                changed_accuracy=evaluate_model_response.difference,
                best_model_path=self.model_eval_config.final_model_file_path)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e