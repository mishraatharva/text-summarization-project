import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts
from src.entity.config_entity import DataTransformationConfig


class TransformationPipeline:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def start_data_transformation(self, data_ingestion_artifacts = DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifacts = data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
    
    
    def run_pipeline(self,data_ingestion_artifacts):
        data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifacts=data_ingestion_artifacts
            )
        return data_transformation_artifacts