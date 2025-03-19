from flask import Flask, render_template, request, jsonify
from chatbot import DialoGPTChatbot
import os

# Create templates directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
os.makedirs(template_dir, exist_ok=True)

# Create Flask app with correct template folder path
app = Flask(__name__, template_folder=template_dir)

# Initialize the chatbot
chatbot = DialoGPTChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if user_message.lower() == 'reset':
        chatbot.reset_chat_history()
        return jsonify({'response': 'Conversation context has been reset.'})
    
    response = chatbot.generate_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)