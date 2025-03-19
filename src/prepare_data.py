import argparse
import os
import json
import random

def format_conversation(conversation, tokenizer_eos_token="<|endoftext|>"):
    """
    Format a conversation for DialoGPT fine-tuning.
    
    Args:
        conversation (list): List of utterances in the conversation.
        tokenizer_eos_token (str): End of text token for the tokenizer.
        
    Returns:
        str: Formatted conversation.
    """
    formatted = ""
    for utterance in conversation:
        formatted += utterance.strip() + tokenizer_eos_token
    return formatted

def main():
    parser = argparse.ArgumentParser(description="Prepare conversation data for DialoGPT fine-tuning")
    parser.add_argument("--input", type=str, required=True,
                        help="Input file or directory containing conversation data")
    parser.add_argument("--output", type=str, default="../data/training_data.txt",
                        help="Output file for formatted conversations")
    parser.add_argument("--format", type=str, choices=["json", "txt"], default="json",
                        help="Format of input data (json or txt)")
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>",
                        help="End of text token for the tokenizer")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    formatted_conversations = []
    
    if args.format == "json":
        # Process JSON format 
        with open(args.input, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
            
        for conversation in conversations:
            formatted = format_conversation(conversation, args.eos_token)
            formatted_conversations.append(formatted)
    
    else:  # txt format
        
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                utterances = line.strip().split('\t')
                formatted = format_conversation(utterances, args.eos_token)
                formatted_conversations.append(formatted)
    
    # Write formatted conversations to output file
    with open(args.output, 'w', encoding='utf-8') as f:
        for conversation in formatted_conversations:
            f.write(conversation + '\n')
    
    print(f"Processed {len(formatted_conversations)} conversations.")
    print(f"Formatted data saved to: {args.output}")

if __name__ == "__main__":
    main()