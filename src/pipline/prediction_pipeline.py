import sys
from typing import List, Union

import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelEvaluationConfig
from src.constants import TARGET_COLUMN


class PredictionPipeline:
    """
    Simple prediction pipeline that loads the final saved model (MyModel)
    and exposes convenience methods you can call from FastAPI (or any other app).

    The saved model is the same object created in `ModelTrainer`:
        final_model = MyModel(preprocessing_object=preprocessor, trained_model_object=trained_model)
    """

    def __init__(self):
        try:
            logging.info("Initializing PredictionPipeline.")
            model_eval_config = ModelEvaluationConfig()
            self._model_path = model_eval_config.final_model_file_path

            logging.info(f"Loading model from path: {self._model_path}")
            self._model = load_object(self._model_path)
            logging.info("Model successfully loaded for prediction.")
        except Exception as e:
            logging.error("Error occurred while initializing PredictionPipeline", exc_info=True)
            raise MyException(e, sys) from e

    def _build_input_dataframe(self, texts: List[str]) -> pd.DataFrame:
        """
        Build a DataFrame consistent with the training schema:
        - 'Title' column for the text feature
        - 'Class Index' (TARGET_COLUMN) added as a dummy column so that
          the stored preprocessing pipeline can run without raising.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            df = pd.DataFrame({"Title": texts})

            # Add dummy target column; MyModel will also guard, but this
            # keeps the shape close to training-time DataFrame.
            if TARGET_COLUMN not in df.columns:
                df[TARGET_COLUMN] = 0

            return df
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, texts: Union[str, List[str]]) -> List[int]:
        """
        Public method to run predictions.

        Parameters
        ----------
        texts : str or List[str]
            Single news title/description or a list of them.

        Returns
        -------
        List[int]
            Predicted class indices for each input text.

        Example (FastAPI usage)
        -----------------------
        pipeline = PredictionPipeline()
        preds = pipeline.predict(["Some news title"])
        """
        try:
            logging.info("Building input DataFrame for prediction.")
            input_df = self._build_input_dataframe(texts=texts)

            logging.info("Running model prediction.")
            preds = self._model.predict(input_df)

            # Ensure result is a plain Python list
            return list(map(int, preds))
        except Exception as e:
            logging.error("Error occurred during prediction", exc_info=True)
            raise MyException(e, sys) from e


__all__ = ["PredictionPipeline"]


