## End-to-End Text Summarization with Fine-Tuned T5 Model

### Overview:

This project demonstrates a complete pipeline for text summarization using the T5 Transformer model, fine-tuned on a custom dataset. 
The T5 model, based on the Transformer architecture, is renowned for its capabilities in sequence-to-sequence tasks, making it ideal for tasks like summarization.
Also, trained sentencepiece tokenizer on custom dataset.

### Key Features:

**Fine-Tuning:** The T5 model is fine-tuned on a specific dataset for abstractive summarization.

**Model Saving:** After fine-tuning, the model and tokenizer are saved locally and gcp.

**Deployment with Flask:** The fine-tuned T5 model is deployed as a web application using Flask, allowing users to input text and receive summarized outputs in real-time.

### Acknowledgments:

This project utilizes the Hugging Face Transformers library, sentencepiece tokenizer and Flask for deployment.