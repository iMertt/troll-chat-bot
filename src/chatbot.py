import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DialoGPTChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium", device=None):
        """
        Initialize the DialoGPT chatbot.
        
        Args:
            model_name (str): The name of the DialoGPT model to use.
            device (str, optional): The device to run the model on ('cuda' or 'cpu').
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading DialoGPT model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # For context management
        self.chat_history_ids = None
        self.max_context_turns = 5
        self.context_turns = 0
    
    def generate_response(self, user_input, max_length=1000):
        """
        Generate a response to the user input.
        
        Args:
            user_input (str): The user's input text.
            max_length (int, optional): Maximum length of the generated response.
            
        Returns:
            str: The chatbot's response.
        """
        # Encode the user input
        input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt").to(self.device)
        
        # Append to chat history if it exists
        if self.chat_history_ids is not None:
            input_ids = torch.cat([self.chat_history_ids, input_ids], dim=-1)
        
        # Generate a response
        chat_history_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7
        )
        
        # Get the response (excluding the input)
        response = self.tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Update chat history
        self.chat_history_ids = chat_history_ids
        self.context_turns += 1
        
        # Reset context if it exceeds max turns
        if self.context_turns >= self.max_context_turns:
            self.reset_chat_history()
        
        return response
    
    def reset_chat_history(self):
        """Reset the chat history."""
        self.chat_history_ids = None
        self.context_turns = 0