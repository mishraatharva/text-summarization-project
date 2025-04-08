from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    raw_data_file_path: str


@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str
    contraction_mapping_path : str
    train_data_path : str
    test_data_path : str
    validation_data_path : str


@dataclass
class T5ModelTrainerArtifacts: 
    trained_model_path:str
    trained_tokenizer_path:str




# @dataclass
# class ModelEvaluationArtifacts:
#     is_model_accepted: bool 