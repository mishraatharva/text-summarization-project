from src.entity.config_entity import DataTransformationConfig, DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts
import os
import logging
from src.exception import *
import sys
import pandas as pd
import re
from bs4 import BeautifulSoup
import json
from nltk.corpus import stopwords
from src.configuration import gcloud_syncer
from sklearn.model_selection import train_test_split
from src.configuration.gcloud_syncer import GCloudSync



class DataTransformation:
    
    def __init__(self,data_transformation_config: DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.stop_words = set(stopwords.words('english')),
        # self.cloud_sync = GCloudSync()

    

    def raw_text_cleaning(self,text_data):
        try:
            newString = text_data.lower()
            newString = BeautifulSoup(newString, "lxml").text
            newString = re.sub(r'\([^)]*\)', '', newString)
            newString = re.sub('"','', newString)
            newString = ' '.join([self.CONTRACTION_MAPPING[t] if t in self.CONTRACTION_MAPPING else t for t in newString.split(" ")])    
            newString = re.sub(r"'s\b","",newString)
            newString = re.sub("[^a-zA-Z]", " ", newString) 
            tokens = [w for w in newString.split() if not w in self.stop_words]
            long_words=[]
            for i in tokens:
                if len(i)>=3:
                    long_words.append(i)
            
            return (" ".join(long_words)).strip()

        except Exception as e:
            raise CustomException(e,sys) from e
    

    def raw_summary_cleaning(self,summary):
        try:

            newString = re.sub('"','', summary)
            newString = ' '.join([self.CONTRACTION_MAPPING[t] if t in self.CONTRACTION_MAPPING else t for t in newString.split(" ")])
            newString = re.sub(r"'s\b","",newString)
            newString = re.sub("[^a-zA-Z]", " ", newString)
            newString = newString.lower()
            tokens=newString.split()
            newString=''
             
            for i in tokens:
                if len(i)>1:                                 
                    newString=newString+i+' '
            return newString
        
        except Exception as e:
            raise CustomException(e,sys) from e
    
        
    def clean_data(self,raw_data):
        logging.info("Entered the clean_data method of Data transformation class and cleaning started.")

        data_copy = pd.DataFrame(raw_data[["Text","Summary"]], columns=["Text","Summary"])

        data_copy = data_copy[data_copy['Summary'].isna() == False]
        
        """Cleaning Text Data"""
        logging.info("Entered into the raw_text_cleaning function")
        cleaned_text = data_copy["Text"].apply(self.raw_text_cleaning)
        logging.info(f"Exited the raw_text_cleaning function and returned the cleaned-text-data")

        """Cleaning Summary Data"""
        logging.info("Entered into the raw_summary_cleaning function")
        cleaned_summary = data_copy["Summary"].apply(self.raw_summary_cleaning)
        logging.info(f"Exited the raw_summary_cleaning function and returned the cleaned-summary-data")

        """Creating Final Cleaned Data"""
        cleaned_data = pd.DataFrame({
            "Text": cleaned_text,
            "Summary": cleaned_summary
        })

        cleaned_data['Summary'] = data_copy['Summary'].apply(lambda x : '_START_ '+ x + ' _END_')
        logging.info(f"Final Data frame created with cleaned Text and Summary data.")

        return cleaned_data
    

    def split_save_to_gcp(self):
        logging.info("Entered the split_save_to_gcp of Data transformation class and saving 'train_data', 'test_data', and 'validation_data'  to gcp.")

        cleaned_data = pd.read_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH).dropna()
        
        x_train,x_test,y_train,y_test=train_test_split(cleaned_data['Text'],cleaned_data['Summary'],test_size=0.20,random_state=42,shuffle=True)
        x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.20,random_state=42,shuffle=True)
        

        train_data = pd.concat([x_train,y_train], axis=1)
        validation_data = pd.concat([x_test,y_test], axis=1)
        test_data = pd.concat([x_val,y_val], axis=1)

        logging.info(f" 'train_data' created with shape:{train_data.shape}")
        logging.info(f" 'test_data' created with shape:{test_data.shape}")
        logging.info(f" 'validation_data' created with shape:{validation_data.shape}")

        train_data.to_csv(self.data_transformation_config.TRAIN_FILE_PATH)
        test_data.to_csv(self.data_transformation_config.TEST_FILE_PATH)
        validation_data.to_csv(self.data_transformation_config.VALIDATION_FILE_PATH)
        logging.info("all 'train_data' and 'test_data' and 'validation_data' saved to {}")




        # GCloudSync.upload_to_gcs(
        #             bucket_name = self.data_transformation_config.BUCKET_NAME,
        #             source_file = train_data,
        #             destination_blob_name  = "train_data.csv",
        #             logger=logging,
        #             chunk_size = 10 * 1024 * 1024
        #             )
        # logging.info("train_data saved to gcp")


        # GCloudSync.upload_to_gcs(
        #             bucket_name = self.data_transformation_config.BUCKET_NAME,
        #             source_file = test_data,
        #             destination_blob_name  = "test_data.csv",
        #             logger=logging,
        #             chunk_size = 10 * 1024 * 1024
        #             )
        # logging.info("test_data saved to gcp")


        # GCloudSync.upload_to_gcs(
        #             bucket_name = self.data_transformation_config.BUCKET_NAME,
        #             source_file = validation_data,
        #             destination_blob_name  = "validation_data.csv",
        #             logger=logging,
        #             chunk_size = 10 * 1024 * 1024
        #             )
        # logging.info("validation_data saved to gcp")


        logging.info("all 'train_data' and 'test_data' and 'validation_data' saved to gcp.")
    
        
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)
            logging.info(" 'Reviews.csv' from raw_data loaded")
            print(self.data_transformation_config.CONTRACTION_MAPPING_PATH)

            with open(self.data_transformation_config.CONTRACTION_MAPPING_PATH, "r") as file:
                self.CONTRACTION_MAPPING = json.load(file)

            cleaned_data = self.clean_data(raw_data)
            
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            cleaned_data.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH,index=False,header=True)

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_data_path = self.data_transformation_config.TRANSFORMED_FILE_PATH,
                contraction_mapping_path = self.data_transformation_config.CONTRACTION_MAPPING_PATH,
                train_data_path=self.data_transformation_config.TRAIN_FILE_PATH,
                test_data_path=self.data_transformation_config.TEST_FILE_PATH,
                validation_data_path=self.data_transformation_config.VALIDATION_FILE_PATH,
            )
            
            self.split_save_to_gcp()
            logging.info("returning the DataTransformationArtifacts")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e