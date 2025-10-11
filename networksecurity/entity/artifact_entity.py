from dataclasses import dataclass
#Artifacts store outputs of each pipeline stage, so downstream stages can consume them.
@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str
#Stores file paths for train and test CSV after ingestion.
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
#Purpose: Artifacts act as contracts between stages of the ML pipeline.