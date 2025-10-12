# Import pipeline components
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation

# Import custom exception and logging utilities
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Import configuration entities
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)
# Optional: Model training imports commented for now
# from networksecurity.components.model_trainer import ModelTrainer
# from networksecurity.entity.config_entity import ModelTrainerConfig

import sys  # For exception handling

if __name__ == '__main__':
    try:
        # Step 1: Initialize the pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # ------------------- DATA INGESTION -------------------
        # Step 2: Initialize data ingestion configuration
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        # Step 3: Create DataIngestion component
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate data ingestion")

        # Step 4: Start data ingestion and get the artifact (paths to train/test files)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        print("Data Ingestion Artifact:", data_ingestion_artifact)

        # ------------------- DATA VALIDATION -------------------
        # Step 5: Initialize data validation configuration
        data_validation_config = DataValidationConfig(training_pipeline_config)

        # Step 6: Create DataValidation component
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiate data validation")

        # Step 7: Run data validation and get validation artifact
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print("Data Validation Artifact:", data_validation_artifact)

        # ------------------- DATA TRANSFORMATION (Optional) -------------------
        # Uncomment when DataTransformation is implemented
        # data_transformation_config = DataTransformationConfig(training_pipeline_config)
        # logging.info("Data transformation started")
        # data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        # data_transformation_artifact = data_transformation.initiate_data_transformation()
        # logging.info("Data transformation completed")
        # print("Data Transformation Artifact:", data_transformation_artifact)

        # ------------------- MODEL TRAINING (Optional) -------------------
        # Uncomment when ModelTrainer is implemented
        # logging.info("Model training started")
        # model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        # model_trainer = ModelTrainer(
        #     model_trainer_config=model_trainer_config,
        #     data_transformation_artifact=data_transformation_artifact
        # )
        # model_trainer_artifact = model_trainer.initiate_model_trainer()
        # logging.info("Model training artifact created")
        # print("Model Trainer Artifact:", model_trainer_artifact)

    except Exception as e:
        # Wrap all exceptions in a custom NetworkSecurityException for consistent error handling
        raise NetworkSecurityException(e, sys)
