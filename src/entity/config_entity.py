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

        self.OUTPUT_DIR = self.TRAINED_MODEL_PATH
        self.EVALUATION_STRATEGY = "epoch"
        self.EVAL_STEPS = 100
        self.LOGGING_STEPS = 100
        self.LOGGING_DIR = os.path.join(ARTIFACTS_DIR,"t5-logs")
        self.SAVE_STRATEGY = "epoch"
        self.LEARNING_RATE = 1e-5
        self.PER_DEVICE_TRAIN_BATCH_SIZE = 16
        self.PER_DEVICE_TRAIN_EVAL_SIZE = 16
        self.WEIGHT_DECAY = 0.01
        self.SAVE_TOTAL_LIMIT = 3
        self.NUM_TRAIN_EPOCHS = 1
        self.PERDICT_WITH_GENERATOR = True
        self.GENERATION_MAX_LENGTH = 150
        self.GENERATION_NUM_BEAMS = 2
        self.LOAD_BEST_MODEL_AT_END = True
        self.METRIC_FOR_BEST_MODEL = "loss"
        self.GREATER_IS_BETTER = False
        self.LOGGING_FIRST_STEP = True

        # self.LOGS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        # self.LOGS_DIR_PATH = os.path.join(self.LOGS_DIR,LOGS_DIR)


@dataclass
class ModelEvaluationConfig: 
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR) 
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR,TRAINED_MODEL_NAME)
    

        self.TRAINED_TOKENIZER_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_TOKENIZER_PATH = os.path.join(self.TRAINED_TOKENIZER_DIR,TRAINED_TOKENIZER_NAME)

        self.BATCH_SIZE = 16

        self.EPOCH = 1


# @dataclass
# class ModelPusherConfig:

#     def __init__(self):
#         self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
#         self.BUCKET_NAME = BUCKET_NAME
#         self.MODEL_NAME = MODEL_NAME
    




