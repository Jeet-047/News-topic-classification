import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.constants import TARGET_COLUMN
from src.exception import MyException
from src.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.yes: int = 0
        self.no: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: Pipeline):
        """
        :param preprocessing_object: Fitted preprocessing pipeline (ColumnTransformer)
        :param trained_model_object: Trained classification pipeline (e.g. TF-IDF + classifier)
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Apply the stored preprocessing pipeline and then predict with the trained model.

        The preprocessing pipeline (ColumnTransformer with ``remainder='passthrough'``)
        was fitted on a DataFrame that contained both the text feature columns and the
        target column (``TARGET_COLUMN``). Because of this, at transform time it expects
        the target column to be present as well.

        To avoid errors like ``columns are missing: {'Class Index'}`` when we only pass
        feature columns at inference/evaluation time, this method:
        - Ensures the ``TARGET_COLUMN`` exists in the DataFrame (adds a dummy column if missing)
        - Runs the preprocessing pipeline
        - Uses only the first transformed column (processed text feature) for prediction,
          which is what the trained model was originally fitted on.
        """
        try:
            logging.info("Starting prediction process in MyModel.")

            df = dataframe.copy()

            # Ensure the target column exists so the ColumnTransformer does not raise
            if TARGET_COLUMN not in df.columns:
                logging.info(
                    f"Input DataFrame missing target column '{TARGET_COLUMN}'. "
                    f"Adding a dummy column so the preprocessing pipeline can run."
                )
                df[TARGET_COLUMN] = 0

            # Apply the same preprocessing used during training
            transformed_feature = self.preprocessing_object.transform(df)

            # Convert to DataFrame so we can easily select the feature column
            transformed_df = pd.DataFrame(transformed_feature)

            # During training, the classifier was trained on the first column only
            x_processed = transformed_df.iloc[:, 0]

            # Perform prediction using the trained model
            logging.info("Using the trained model to get predictions.")
            predictions = self.trained_model_object.predict(x_processed)

            return predictions

        except Exception as e:
            logging.error("Error occurred in MyModel.predict", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"