> Note: English is recommended for better interaction with the chatbot.

# DialoGPT Troll Chatbot

A fun and interactive chatbot built with DialoGPT, featuring a modern web interface. The chatbot is designed to engage in playful and entertaining conversations.

## Features

- 🤖 Powered by Microsoft's DialoGPT model
- 💬 Real-time chat interface
- 🎨 Modern and responsive UI design
- ⚡ Fast response times
- 🔄 Conversation reset capability
- 💻 Both CLI and Web interfaces

## Installation

1. Clone the repository:

```bash
git clone https://github.com/iMertt/ai-bot.git
cd ai-bot
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Run the chatbot with the web interface:

```bash
python main.py --mode=web
```

Then open your browser and navigate to: http://127.0.0.1:5000

### Command Line Interface

Run the chatbot in CLI mode:

```bash
python main.py --mode=cli
```

## Configuration Options

You can configure the following options when running the chatbot:

- --mode : Choose between 'cli' or 'web' interface (default: cli)
- --model : Specify the DialoGPT model size:
  - microsoft/DialoGPT-small
  - microsoft/DialoGPT-medium (default)
  - microsoft/DialoGPT-large
- --device : Choose 'cuda' for GPU or 'cpu' for CPU processing
  Example:

```bash
python main.py --mode=web --model=microsoft/DialoGPT-large --device=cuda
```

````

## Project Structure
```plaintext
ai-bot/
├── main.py           # Main entry point
├── requirements.txt  # Project dependencies
├── src/
│   ├── app.py       # Web interface implementation
│   ├── cli.py       # CLI interface implementation
│   └── chatbot.py   # Core chatbot functionality
└── templates/
    └── index.html   # Web interface template
````

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements.

## Author

[M.D.](https://github.com/iMertt)
