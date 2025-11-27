import os
import sys
import pandas as pd
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

from src.constants import SCHEMA_FILE_PATH, DATA_TRANSFORMATION_EMOJI_PATTERN, DATA_TRANSFORMATION_CONTRACTION_MAP
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, read_yaml_file, download_nltk_package_if_needed


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self._text_columns = self._schema_config.get("object_column", [])
            if not self._text_columns:
                raise ValueError("No text columns defined in schema.yaml under object_column.")
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
    
    # Download the NLTK packages if needed
    download_nltk_package_if_needed("punkt", "tokenizers")
    download_nltk_package_if_needed("stopwords", "corpora")

    
    def clean_and_normalize_text(self, text: str) -> str:
        """
        This function clean the text by removing HTML tags, Punctuation, Numbers, 
        Special character, White space, Lower case, Tokenization, Stop word, Remove Contraction, 
        and normalize the text by Lemmatization.
        """
        if pd.isna(text):
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        # Initialize the stop-words
        stop_words = set(stopwords.words("english"))

        text = re.sub(r'<[^>]+>', ' ', text)    # For HTML tgs
        text = re.sub(r"(?!\B'\b)[%s]" % re.escape(string.punctuation.replace("'", "")), " ", text) # For punctuation
        text = re.sub(r'\d+', ' ', text)    # For numbers/digits
        text = DATA_TRANSFORMATION_EMOJI_PATTERN.sub(r"", text) # For emojis
        text = re.sub(' +', ' ', text).strip()  # For white space
        text = text.lower() # Lowercase
        text = text.split() # Tokenization
        text = [word for word in text if word not in stop_words]    # For stop-words
        # Remove Contraction
        clean_tokens = []
        for token in text:
            if token not in DATA_TRANSFORMATION_CONTRACTION_MAP:
                clean_tokens.append(token)
        # Lemmatization
        normalized_tokens = [WordNetLemmatizer().lemmatize(word) for word in clean_tokens]

        return " ".join(normalized_tokens)

    def apply_preprocessing(self, X) -> pd.DataFrame:
        """
        Wraps the single-string function so it can be applied to a Pandas Series (X).
        FunctionTransformer passes the entire column here.
        """
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        else:
            series = pd.Series(X)
        processed_series = series.map(self.clean_and_normalize_text)
        return pd.DataFrame(processed_series, columns=[series.name])


    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including cleaning and normalization of text.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Create the transformer: Wrap the clean_and_normalize_text function
            normalize_transformer = FunctionTransformer(func=self.apply_preprocessing, validate=False)
            # Apply ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('preprocess_text', normalize_transformer, self._text_columns) # Apply to text columns
                ],
                remainder='passthrough' # Keep other columns as they are
            )
            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor text")

            logging.info("Initializing transformation for Training-data")
            train_df_processed = pd.DataFrame(preprocessor.fit_transform(train_df), columns=train_df.columns[::-1])
            logging.info("Initializing transformation for Testing-data")
            test_df_processed = pd.DataFrame(preprocessor.transform(test_df), columns=test_df.columns[::-1])

            logging.info("Saving transformation object and transformed files.")
            # Ensure output directories exist before writing files
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            train_df_processed.to_csv(self.data_transformation_config.transformed_train_file_path,index=False,header=True)
            test_df_processed.to_csv(self.data_transformation_config.transformed_test_file_path,index=False,header=True)
            
            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e