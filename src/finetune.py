import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import argparse
import os

def prepare_dataset(file_path, tokenizer, block_size=128):
    """
    Prepare a dataset for fine-tuning.
    
    Args:
        file_path (str): Path to the text file containing conversations.
        tokenizer: The tokenizer to use.
        block_size (int): The block size for the dataset.
        
    Returns:
        TextDataset: The prepared dataset.
    """
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DialoGPT model")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium",
                        help="Base model to fine-tune")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the training data file")
    parser.add_argument("--output", type=str, default="../models/fine-tuned-dialogpt",
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Block size for the dataset")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    
    args = parser.parse_args()
    
   
    os.makedirs(args.output, exist_ok=True)
    
    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Add special tokens
    special_tokens = {"pad_token": "<PAD>"}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare dataset
    print(f"Preparing dataset from: {args.data}")
    train_dataset = prepare_dataset(args.data, tokenizer, args.block_size)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # not using masked language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10000,
        save_total_limit=2,
        learning_rate=args.learning_rate,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    print(f"Saving model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()