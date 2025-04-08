import os

from datetime import datetime
#####################################################################################################################################################
#### Common constants
#####################################################################################################################################################

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = "artifacts"
BUCKET_NAME = "text-summarization-bucket"
ZIP_FILE_NAME = "Reviews.zip"
LABEL = 'Text'
TWEET = 'Summarize'
GOOGLE_APPLICATION_CREDENTIALS = "radiant-anchor-455909-e0-827ca3c48aed.json"

#####################################################################################################################################################
#### Data ingestion constants
#####################################################################################################################################################

DATA_INGESTION_ARTIFACTS_DIR = "raw_data"
# DATA_INGESTION_IMBALANCE_DATA_DIR = "Reviews.csv"
DATA_INGESTION_RAW_DATA_DIR = "Reviews.csv"

#####################################################################################################################################################
#### Data transformation constants 
#####################################################################################################################################################
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'processed_data'
TRANSFORMED_FILE_NAME = "cleaned_reviews.csv"

TRAIN_FILE_PATH = "train_data.csv"
TEST_FILE_PATH = "test_data.csv"
VALIDATION_FILE_PATH  = "validation_data.csv"

DATA_DIR = "data"
ID = 'id'
AXIS = 1
INPLACE = True
DROP_COLUMNS = ["ProductId", "UserId", "ProfileName", "HelpfulnessNumerator","HelpfulnessDenominator","Score","Time"]
CLASS = 'class'


#####################################################################################################################################################
#### SENTENCEPIECE training constants
#####################################################################################################################################################




#####################################################################################################################################################
#### T5-Model training constants
#####################################################################################################################################################

MODEL_TRAINER_ARTIFACTS_DIR = 't5_model'

# TRAINED_MODEL_DIR = 'trained_model'
TRAINED_MODEL_NAME = 't5-model'

# TRAINED_MODEL_DIR = 'trained_tokenizer'
TRAINED_TOKENIZER_NAME = 't5-tokenizer'
LOGS_DIR = "training_logs"

# TRAINED_MODEL_DIR = 'tokenizer-text'
TEXT_FILE_NAME = 'text'


# RANDOM_STATE = 42
# EPOCH = 1
# BATCH_SIZE = 128
# VALIDATION_SPLIT = 0.2


# # Model Architecture constants
# MAX_WORDS = 50000
# MAX_LEN = 300
# LOSS = 'binary_crossentropy'
# METRICS = ['accuracy']
# ACTIVATION = 'sigmoid'


# # Model  Evaluation constants
# MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
# BEST_MODEL_DIR = "best_Model"
# MODEL_EVALUATION_FILE_NAME = 'loss.csv'


# MODEL_NAME = 'model.h5'
# APP_HOST = "0.0.0.0"
# APP_PORT = 8080
