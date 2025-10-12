# Import necessary entities and utilities
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp  # For statistical test of data drift
import pandas as pd  # For DataFrame operations
import os, sys  # For OS path and exception handling
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    """
    Class to validate ingested data:
    - Checks if dataset columns match the schema
    - Detects data drift between train and test datasets
    - Generates validation artifacts
    """

    # Constructor: initializes with data ingestion artifact and validation config
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            # Store the artifacts and config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            # Load schema file to get expected columns
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            # Wrap any error in a custom exception
            raise NetworkSecurityException(e, sys)

    # Static method to read CSV into DataFrame
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # Validate that the dataset has the expected number of columns
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # Expected column count from schema
            number_of_columns = len(self._schema_config)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")

            # Return True if column count matches schema
            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # Detect data drift using KS test
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True  # Assume no drift initially
            report = {}  # Store drift results for each column

            # Iterate through each column in the dataset
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                # Perform Kolmogorov-Smirnov test
                is_same_dist = ks_2samp(d1, d2)

                # If p-value < threshold, drift is detected
                drift_found = is_same_dist.pvalue < threshold
                if drift_found:
                    status = False  # Overall status = False if any drift found

                # Store p-value and drift status for this column
                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": drift_found
                }

            # Path to save the drift report YAML
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)

            # Write the drift report to YAML
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status  # Return overall status (True if no drift, False if drift)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # Main method to run data validation
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # Get file paths from data ingestion artifact
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read train and test datasets
            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)

            # Validate column counts
            if not self.validate_number_of_columns(train_dataframe):
                logging.error("Train dataframe does not contain all required columns.")
            if not self.validate_number_of_columns(test_dataframe):
                logging.error("Test dataframe does not contain all required columns.")

            # Detect data drift between train and test
            status = self.detect_dataset_drift(train_dataframe, test_dataframe)

            # Ensure directories exist to save validated files
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)

            # Save validated train and test datasets
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            # Return validation artifact with paths and status
            return DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
