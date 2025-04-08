import os
import sys
from zipfile import ZipFile
from src.logger import logging
from src.exception import CustomException
from src.configuration.gcloud_syncer import GCloudSync
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifacts
from pathlib import Path

class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        # self.gcloud = GCloudSync()


    def get_data_from_gcloud(self) -> None:
        try:
            logging.info("Entered the get_data_from_gcloud method of Data ingestion class")
            os.makedirs(Path(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR).parent, exist_ok=True)

            logging.info(f"directory created>>>>>>>{self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR}")


            GCloudSync.download_from_gcs(
                logging,
                bucket_name = self.data_ingestion_config.BUCKET_NAME,                                  # "text-summarization-bucket",
                source_blob_name= self.data_ingestion_config.ZIP_FILE_NAME,                            # "Reviews.zip",  # The object name in the bucket
                destination_file_path = str(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR),  # r"U:/nlp_project/text_summarization/artifacts/Reviews.zip"  # Where to save locally
            )

            logging.info("Exited the download_from_gcs method of Data ingestion class")
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try: 
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return os.path.join(self.data_ingestion_config.ZIP_FILE_DIR, "Reviews.csv")

        except Exception as e:
            raise e

    
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")

        try:
            self.get_data_from_gcloud()
            logging.info("Fetched the data from gcloud bucket")
            raw_data_file_path = self.unzip_and_clean()
            
            logging.info("Unzipped file and split into train and valid")

            data_ingestion_artifacts = DataIngestionArtifacts(
                raw_data_file_path = raw_data_file_path
            )

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e