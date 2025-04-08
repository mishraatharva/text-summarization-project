import os
from src.entity.artifacts_entity import DataTransformationArtifacts
from src.entity.config_entity import T5ModelTrainerConfig, SentencePieceTrainerConfig
import logging
import pandas as pd
import sentencepiece as spm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import sys
from datasets import load_dataset
from src.configuration.gcloud_syncer import GCloudSync
from datasets import load_dataset
import evaluate


class T5ModelTraining:
    def __init__(self,model_training_config: T5ModelTrainerConfig, sentence_piece_config: SentencePieceTrainerConfig ,data_transformation_artifacts:DataTransformationArtifacts):
        self.model_training_config = model_training_config
        self.sentence_piece_config = sentence_piece_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.cloud_sync = GCloudSync()
        self.rouge = evaluate.load("rouge")



    def create_tokenizer(self):
        """Create and save tokinizer for future use"""
        try:
            data = pd.read_csv(self.data_transformation_artifacts.transformed_data_path)["Text"].dropna()
        
            path_to_text = os.path.join(self.model_training_config.TOKENIZER_TEXT_PATH,"train_data.txt")

            with open(path_to_text, "w") as file:
                for str in list(data):
                    file.write(str + "\n")

            spm.SentencePieceTrainer.train(
                    input = self.sentence_piece_config.INPUT,
                    model_prefix = self.sentence_piece_config.MODEL_PREFIX,
                    vocab_size = self.sentence_piece_config.VOCAB_SIZE,
                    character_coverage = self.sentence_piece_config.CHARACTER_COVERAGE,
                    model_type = self.sentence_piece_config.MODEL_TYPE,
                    user_defined_symbols = self.sentence_piece_config.USER_DEFINED_SYMBOLS
            )
        
        except Exception as e:
            print(e)

        logging.info("Tokenizer training completed! Files (t5-tokenizer.model and t5-tokenizer.vocab) saved at: {self.sentence_piece_config.MODEL_PREFIX}")
        
        model_path = self.sentence_piece_config.MODEL_PATH
        final_model_path = os.path.join(self.sentence_piece_config.FINAL_MODEL_PATH)

        logging.info(model_path)
        logging.info(final_model_path)

        tokenizer = T5Tokenizer(vocab_file=model_path)
        tokenizer.save_pretrained(final_model_path)
         
        """Impliment code to push tokenizer to gcp in future"""

        logging.info(f"Tokenizer saved in huggingface format and saved at : {self.sentence_piece_config.MODEL_PREFIX} with name 't5_tokenizer_hf'.")


    def compute_metrics(self,pred):
        pred_ids = pred.predictions
        labels_ids = pred.label_ids

        # Ensure predictions are within tokenizer vocab size
        vocab_size = self.tokenizer.vocab_size
        pred_ids = [[token if token < vocab_size else self.tokenizer.unk_token_id for token in seq] for seq in pred_ids]

        # Decode the predictions and labels
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # Compute metric (replace with actual metric like ROUGE)
        return self.rouge.compute(predictions=pred_str, references=label_str)
    

    def create_dataset(self):

        logging.info("inside create_dataset of T5ModelTraining class")

        dataset = load_dataset("csv", data_files={
                        "train": self.data_transformation_artifacts.train_data_path, 
                        "test": self.data_transformation_artifacts.test_data_path,
                        "validation": self.data_transformation_artifacts.validation_data_path}
                        )
        
        train_subset = dataset["train"].select(range(1000))
        validation_subset = dataset["validation"].select(range(1000))
        test_subset = dataset["test"].select(range(1000))
        logging.info("final 'train_sunset', 'test_subset' and 'validation_subset' created to 1k data points each")
        return train_subset, test_subset, validation_subset
    
    
    
    def preprocess_function(self,examples):
        inputs = ["summarize: " + doc for doc in examples["Text"]]
        model_inputs = self.tokenizer(inputs, max_length=500, truncation=True, padding=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["Summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def train_t5_model(self):
        """Train T5-model"""
        logging.info("inside train_t5_model of T5ModelTraining class")

        train_subset, test_subset, validation_subset = self.create_dataset()
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.sentence_piece_config.FINAL_MODEL_PATH)
        logging.info("loaded -> saved tokenizer")

        tokenized_train = train_subset.map(self.preprocess_function, batched=True)
        logging.info("train_subset tokenization complete")

        tokenized_validation = validation_subset.map(self.preprocess_function, batched=True)
        logging.info("validation_subset tokenization complete")

        # tokenized_test = test_subset.map(self.preprocess_function, batched=True)
        # logging.info("test_subset tokenization complete")

        data_collator = DataCollatorForSeq2Seq(self.tokenizer)


        training_args = Seq2SeqTrainingArguments(
            output_dir = self.model_training_config.output_dir,
            evaluation_strategy = self.model_training_config.evaluation_strategy,
            eval_steps = self.model_training_config.eval_steps,
            logging_steps = self.model_training_config.logging_steps,
            logging_dir = self.model_training_config.logging_dir,
            report_to = self.model_training_config.report_to,
            save_strategy = self.model_training_config.save_strategy,
            learning_rate = self.model_training_config.learning_rate,
            per_device_train_batch_size = self.model_training_config.per_device_train_batch_size,
            per_device_eval_batch_size = self.model_training_config.per_device_eval_batch_size,
            weight_decay = self.model_training_config.weight_decay,
            save_total_limit = self.model_training_config.save_total_limit,
            num_train_epochs = self.model_training_config.num_train_epochs,
            predict_with_generate = self.model_training_config.predict_with_generate,
            generation_max_length = self.model_training_config.generation_max_length,
            generation_num_beams = self.model_training_config.generation_num_beams,
            load_best_model_at_end = self.model_training_config.load_best_model_at_end,
            metric_for_best_model = self.model_training_config.metric_for_best_model,
            greater_is_better = self.model_training_config.greater_is_better,
            logging_first_step = self.model_training_config.logging_first_step,
            # label_smoothing_factor=0.1
            )
        
        trainer = Seq2SeqTrainer(
                model=T5ForConditionalGeneration.from_pretrained('t5-small'),
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_validation,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
        
        trainer.train()



    def initiate_model_training(self):
        try:
            os.makedirs(self.model_training_config.TRAINED_MODEL_PATH,exist_ok=True)
            os.makedirs(self.model_training_config.TRAINED_TOKENIZER_PATH,exist_ok=True)
            os.makedirs(self.model_training_config.TOKENIZER_TEXT_PATH,exist_ok=True)
            # os.makedirs(self.model_training_config.LOGS_DIR_PATH,exist_ok=True)

            logging.info(f"path to save t5-model at {self.model_training_config.TRAINED_MODEL_PATH}")
            logging.info(f"path to save t5-model at {self.model_training_config.TRAINED_TOKENIZER_PATH}")
            logging.info(f"path to save t5-model at {self.model_training_config.TOKENIZER_TEXT_PATH}")
            # logging.info(f"path to save t5-model at {self.model_training_config.LOGS_DIR_PATH}")
            # print(os.path.join(self.sentence_piece_config.VOCAB_PATH))
            

            self.create_tokenizer()

        except Exception as e:
            print(e)