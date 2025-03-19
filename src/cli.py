import argparse
from chatbot import DialoGPTChatbot

def main():
    parser = argparse.ArgumentParser(description="DialoGPT Chatbot CLI")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium", 
                        help="DialoGPT model to use (small, medium, large)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run the model on (cuda or cpu)")
    args = parser.parse_args()
    
    print("Initializing DialoGPT Chatbot...")
    chatbot = DialoGPTChatbot(model_name=args.model, device=args.device)
    
    print("\nChatbot is ready! Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'reset' to reset the conversation context.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        if user_input.lower() == "reset":
            chatbot.reset_chat_history()
            print("Chatbot: Conversation context has been reset.")
            continue
        
        response = chatbot.generate_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()