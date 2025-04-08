from src.pipeline.stage_01_ingestion_pipeline import IngestionPipeline
from src.pipeline.stage_02_transformation_pipeline import TransformationPipeline
from src.pipeline.stage_03_training_pipeline import T5TrainingPipeline

from src.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts

from src.constants import *
from src.logger import logging
from google.oauth2 import service_account
from google.cloud import storage
from dotenv import load_dotenv
import os
import sklearn
import pandas as pd
import warnings

load_dotenv()

logging.info("Loading credentials from environment variables")
# credentials_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

credentials_path = os.path.join(os.getcwd(),GOOGLE_APPLICATION_CREDENTIALS)
project_id = os.environ['GCP_PROJECT_ID']
bucket_name = os.environ['GCP_BUCKET_NAME']

logging.info(credentials_path)
logging.info(project_id)
logging.info(bucket_name)

# Create a credentials object
credentials = service_account.Credentials.from_service_account_file(credentials_path)

pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")


###############################
# start model training
###############################

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# try:
#     stage = "INGESTION STAGE"
#     logging.info(F">>>>>>>>>>>>>>>>{stage}-STARTED<<<<<<<<<<<<<<<<<")
#     training_pipeline = IngestionPipeline()
#     data_ingestion_artifacts = training_pipeline.run_pipeline()
#     logging.info(F">>>>>>>>>>>>>>>>{stage}-COMPLITED<<<<<<<<<<<<<<<<<")
# except Exception as e:
#     print(e)



# try:
#     data_ingestion_artifacts = DataIngestionArtifacts(raw_data_file_path = r"U:\nlp_project\text_summarization\artifacts\row_data\Reviews.csv")
#     stage = "TRANSFORMATION STAGE"
#     logging.info(F">>>>>>>>>>>>>>>>{stage}-STARTED<<<<<<<<<<<<<<<<<")
#     training_pipeline = TransformationPipeline()
#     data_transformation_artifacts = training_pipeline.run_pipeline(data_ingestion_artifacts)
#     logging.info(F">>>>>>>>>>>>>>>>{stage}-COMPLITED<<<<<<<<<<<<<<<<<")
# except Exception as e:
#     print(e)



try:
    data_transformation_artifacts = DataTransformationArtifacts(transformed_data_path = r"U:\nlp_project\text_summarization\artifacts\processed_data\cleaned_reviews.csv",
                                                           contraction_mapping_path = r"U:\nlp_project\text_summarization\artifacts\contraction_data\CONTRACTION_MAPPING.json",
                                                           train_data_path = r"U:\nlp_project\text_summarization\artifacts\processed_data\train_data.csv",
                                                           test_data_path = r"U:\nlp_project\text_summarization\artifacts\processed_data\test_data.csv",
                                                           validation_data_path = r"U:\nlp_project\text_summarization\artifacts\processed_data\validation_data.csv")
    stage = "TRAINING STAGE"
    logging.info(F">>>>>>>>>>>>>>>>{stage}-STARTED<<<<<<<<<<<<<<<<<")
    training_pipeline = T5TrainingPipeline()
    training_pipeline.run_pipeline(data_transformation_artifacts)
    logging.info(F">>>>>>>>>>>>>>>>{stage}-COMPLITED<<<<<<<<<<<<<<<<<")
except Exception as e:
    print(e)