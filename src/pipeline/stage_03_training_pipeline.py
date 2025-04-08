from src.entity.artifacts_entity import T5ModelTrainerArtifacts, DataTransformationArtifacts
from src.entity.config_entity import T5ModelTrainerConfig, SentencePieceTrainerConfig
import logging
from src.components.t5_model_training import T5ModelTraining

class T5TrainingPipeline:
    def __init__(self):
        self.model_training_config = T5ModelTrainerConfig()
        self.sentence_piece_config = SentencePieceTrainerConfig()
    


    def start_model_training(self, data_transformation_artifacts = DataTransformationArtifacts):
        logging.info("Entered the start_model_training method of T5TrainingPipeline class")
        try:
            t5_model_training = T5ModelTraining(self.model_training_config, self.sentence_piece_config ,data_transformation_artifacts)
            model_training_artifacts = t5_model_training.initiate_model_training()
            return model_training_artifacts
        except Exception as e:
            print(e)


    def run_pipeline(self, data_transformation_artifacts) -> T5ModelTrainerArtifacts:
        model_training_artifacts = self.start_model_training(
                data_transformation_artifacts=data_transformation_artifacts
            )
        return model_training_artifacts