import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="DialoGPT Chatbot")
    parser.add_argument("--mode", type=str, choices=["cli", "web"], default="cli",
                        help="Run mode: cli for command line interface, web for web interface")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium",
                        help="DialoGPT model to use (small, medium, large)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run the model on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Add src directory to path
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    
    if args.mode == "cli":
        from src.cli import main as cli_main
        sys.argv = [sys.argv[0]] + [f"--model={args.model}"]
        if args.device:
            sys.argv.append(f"--device={args.device}")
        cli_main()
    else:
        from src.app import app
        app.run(debug=True)

if __name__ == "__main__":
    main()