import mlflow
import mlflow.pytorch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
import logging
from datasets import load_dataset
from src.entity.config_entity import ModelEvaluationConfig
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

# mlflow.set_experiment("T5_Summarization_Experiment")

class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.references = tokenized_data["Summary"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "reference": self.references[idx]
            }
        

class T5ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    
    def create_dataset(self,data_transformation_artifacts):

        logging.info("inside create_dataset of T5ModelTraining class")

        dataset = load_dataset("csv", data_files={
            "test": data_transformation_artifacts.test_data_path,
            })
        
        eval_dataset= dataset["test"].select(range(1000))
        logging.info(" 'eval_dataset' created to 1k data points.")
        return eval_dataset


    def preprocess_function(self,examples):
        inputs = ["summarize: " + doc for doc in examples["Text"]]
        model_inputs = self.tokenizer(inputs, max_length=500, truncation=True, padding=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["Summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def evaluate_t5_model(self,data_transformation_artifacts,model_training_artifacts):
        logging.info("Entered the evaluate_t5_model method of T5ModelEvaluation class")

        with mlflow.start_run():
            mlflow.log_params({
                "batch_size": self.model_evaluation_config.BATCH_SIZE,
                "epochs": self.model_evaluation_config.EPOCH,
            })

        model = T5ForConditionalGeneration.from_pretrained(model_training_artifacts.trained_model_path)
        logging.info(f"model loaded")


        self.tokenizer = T5Tokenizer.from_pretrained(model_training_artifacts.trained_tokenizer_path)
        logging.info("tokenizer loaded")


        eval_dataset = self.create_dataset(data_transformation_artifacts)
        logging.info("'eval_dataset' loaded")

        tokenized_eval = eval_dataset.map(self.preprocess_function, batched=True)
        logging.info("'eval_dataset' tokenized")

        dataset = SummarizationDataset(tokenized_eval)
        dataloader = DataLoader(dataset, batch_size=16)

        predictions = []
        references = []

        for batch in tqdm(dataloader, desc="Generating summaries"):
            with torch.no_grad():
                input_ids = batch["input_ids"].to("cpu")
                attention_mask = batch["attention_mask"].to("cpu")
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150)
                decoded_preds = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                predictions.extend(decoded_preds)
                references.extend(batch["reference"])
        
        logging.info("Predictions generated for all batches")


        # Load and compute ROUGE
        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        rouge_score = result["rouge2"]


        logging.info(f"ROUGE-2 Score: {rouge_score}")
        mlflow.log_metric("rouge2", rouge_score)
        mlflow.pytorch.log_model(model, "model")

        logging.info("Model evaluation and logging completed")

        
    def initiate_t5_model_evaluation(self,data_transformation_artifacts,model_training_artifacts):
        logging.info("Entered the initiate_t5_model_evaluation method of T5ModelEvaluation class")

        self.evaluate_t5_model(data_transformation_artifacts,model_training_artifacts)