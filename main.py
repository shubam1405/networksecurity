import sys
import dagshub
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig
)

# ✅ Initialize DagsHub once (safe placement)
try:
    dagshub.init(repo_owner='shubam1405', repo_name='networksecurity', mlflow=True)
    print("✅ DagsHub MLflow tracking initialized successfully.")
except Exception as e:
    print(f"⚠️ Could not initialize DagsHub MLflow tracking: {e}")

if __name__ == '__main__':
    try:
        # ------------------------
        # 1️⃣ Data Ingestion
        # ------------------------
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)

        logging.info("🚀 Initiating data ingestion...")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        logging.info("✅ Data ingestion completed.")

        # ------------------------
        # 2️⃣ Data Validation
        # ------------------------
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)

        logging.info("🧩 Initiating data validation...")
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("✅ Data validation completed.")

        # ------------------------
        # 3️⃣ Data Transformation
        # ------------------------
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)

        logging.info("🔄 Initiating data transformation...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("✅ Data transformation completed.")

        # ------------------------
        # 4️⃣ Model Training
        # ------------------------
        logging.info("🏋️ Initiating model training...")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )

        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("✅ Model training completed successfully.")
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
