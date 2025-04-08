import os
from google.cloud import storage

class GCloudSync:

    def upload_to_gcs(bucket_name, source_file, destination_blob_name, logger ,chunk_size):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Enable resumable uploads with a chunk size
        blob.chunk_size = chunk_size  

        try:
            csv_data = source_file.to_csv(index=False)
            
            blob.upload_from_string(csv_data)

            logger.info("File csv_data uploaded to gs://{bucket_name}/{destination_blob_name}")
        
        except Exception as e:
            logger.info(f"Upload failed: {e}")


    def download_from_gcs(logging, bucket_name, source_blob_name, destination_file_path):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        logging.info(f"client >>>> {client}")
        logging.info(f"bucket >>>> {bucket}")
        logging.info(f"blob >>>> {blob}")

        try:
            blob.download_to_filename(destination_file_path)
            logging.info(f"File gs://{bucket_name}/{source_blob_name} downloaded to {destination_file_path}")
            logging.info(f"Downloaded data save to: {destination_file_path}")
        except Exception as e:
            logging.info(f"Download failed: {e}")