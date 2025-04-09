from src.entity.artifacts_entity import ModelEvaluationArtifacts, T5ModelTrainerArtifacts, DataTransformationArtifacts
from src.entity.config_entity import ModelEvaluationConfig
import logging
from src.components.t5_model_evaluation import T5ModelEvaluation

class T5EvaluationPipeline:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def start_model_evaluation(self,data_transformation_artifacts,model_training_artifacts):
        logging.info("Entered the start_model_evaluation method of T5EvaluationPipeline class")
        try:
            t5_model_evaluation = T5ModelEvaluation()
            model_training_artifacts = t5_model_evaluation.initiate_t5_model_evaluation(data_transformation_artifacts,model_training_artifacts)
            return model_training_artifacts
        except Exception as e:
            print(e)

    def run_pipeline(self, data_transformation_artifacts: DataTransformationArtifacts, model_training_artifacts : T5ModelTrainerArtifacts) -> ModelEvaluationArtifacts:
        model_evaluation_artifacts = self.start_model_evaluation(data_transformation_artifacts,model_training_artifacts)
        return model_evaluation_artifacts