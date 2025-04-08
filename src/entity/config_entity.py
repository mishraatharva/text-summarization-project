from dataclasses import dataclass
from src.constants import *
import os
from pathlib import Path

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.RAW_DATA_PATH = Path
        self.BUCKET_NAME = BUCKET_NAME
        self.ZIP_FILE_NAME = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR = str(Path.cwd() / "artifacts" / "row_data" / ZIP_FILE_NAME)
        # self.DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATA_INGESTION_IMBALANCE_DATA_DIR)
        self.NEW_DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATA_INGESTION_RAW_DATA_DIR)
        self.ZIP_FILE_DIR = Path.cwd() / "artifacts" / "row_data"
        self.ZIP_FILE_PATH = self.DATA_INGESTION_ARTIFACTS_DIR


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRANSFORMED_FILE_NAME)

        self.TRAIN_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRAIN_FILE_PATH)
        self.TEST_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TEST_FILE_PATH)
        self.VALIDATION_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,VALIDATION_FILE_PATH)

        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE 
        self.DROP_COLUMNS = DROP_COLUMNS
        self.CONTRACTION_MAPPING_PATH = os.path.join(os.getcwd(), ARTIFACTS_DIR,"contraction_data","CONTRACTION_MAPPING.json")
        self.BUCKET_NAME = BUCKET_NAME


@dataclass
class SentencePieceTrainerConfig:
    def __init__(self):
        self.TOKENIZER_TEXT_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        self.INPUT : str = os.path.join(self.TOKENIZER_TEXT_DIR,TEXT_FILE_NAME, "train_data.txt")

        self.MODEL_PREFIX : str = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_TOKENIZER_NAME, "t5-tokenizer")
        self.MODEL_PATH = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_TOKENIZER_NAME, "t5-tokenizer.model")


        self.FINAL_MODEL_PATH : str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_TOKENIZER_NAME, 't5_tokenizer_hf')

        self.VOCAB_SIZE : int = 32000
        self.CHARACTER_COVERAGE : float= 0.9995
        self.MODEL_TYPE : str = "unigram"
        self.USER_DEFINED_SYMBOLS : list = ["[PAD]"]


@dataclass
class T5ModelTrainerConfig: 
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR) 
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR,TRAINED_MODEL_NAME)

        self.TRAINED_TOKENIZER_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_TOKENIZER_PATH = os.path.join(self.TRAINED_TOKENIZER_DIR,TRAINED_TOKENIZER_NAME)

        self.TOKENIZER_TEXT_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        self.TOKENIZER_TEXT_PATH = os.path.join(self.TOKENIZER_TEXT_DIR,TEXT_FILE_NAME)

        self.output_dir="./results",
        self.evaluation_strategy="epoch",
        self.eval_steps=100,
        self.logging_steps=100,
        self.logging_dir="./logs",
        self.report_to="all",
        self.save_strategy="epoch",
        self.learning_rate=1e-5,
        self.per_device_train_batch_size=16,
        self.per_device_eval_batch_size=16,
        self.weight_decay=0.01,
        self.save_total_limit=3,
        self.num_train_epochs=3,
        self.predict_with_generate=True,
        self.generation_max_length=150,
        self.generation_num_beams=6,
        self.load_best_model_at_end=True,
        self.metric_for_best_model="loss",
        self.greater_is_better=False,
        self.logging_first_step=True,

        # self.LOGS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        # self.LOGS_DIR_PATH = os.path.join(self.LOGS_DIR,LOGS_DIR)




# @dataclass
# class ModelEvaluationConfig: 
#     def __init__(self):
#         self.MODEL_EVALUATION_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
#         self.BEST_MODEL_DIR_PATH: str = os.path.join(self.MODEL_EVALUATION_MODEL_DIR,BEST_MODEL_DIR)
#         self.BUCKET_NAME = BUCKET_NAME 
#         self.MODEL_NAME = MODEL_NAME 



# @dataclass
# class ModelPusherConfig:

#     def __init__(self):
#         self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
#         self.BUCKET_NAME = BUCKET_NAME
#         self.MODEL_NAME = MODEL_NAME
    




