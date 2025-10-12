import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

# Import constants and configuration details
from networksecurity.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

# Import custom entities and utilities
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    """
    This class handles the transformation of training and testing datasets.
    It applies imputation for missing values using KNNImputer and saves
    the transformed data and transformation objects as artifacts.
    """

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Constructor initializes paths for validated data and configuration details
        for saving transformed outputs.
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads a CSV file and returns it as a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Initializes a KNNImputer using the parameters specified in constants
        and returns a sklearn Pipeline containing this imputer.
        """
        logging.info("Creating KNNImputer pipeline for data transformation.")
        try:
            # Initialize KNNImputer with configured parameters
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"KNNImputer initialized with params: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            # Create pipeline (can add more preprocessing steps later if needed)
            processor = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes the complete data transformation process:
        - Reads validated train and test data
        - Applies KNN imputation
        - Saves transformed arrays and preprocessor object
        """
        logging.info("Starting data transformation process.")
        try:
            # Step 1: Read validated train and test datasets
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Step 2: Separate input and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)  # Convert -1 to 0

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Step 3: Get preprocessing pipeline (KNNImputer)
            preprocessor = self.get_data_transformer_object()

            # Step 4: Fit on training data and transform both train and test
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Step 5: Combine transformed input features with target column
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Step 6: Save transformed arrays and preprocessor
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # Also save preprocessor for final model use
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # Step 7: Create and return artifact containing file paths
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("Data transformation completed successfully.")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
