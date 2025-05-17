# Installation Guide: HER2 QA Chatbot with Ollama

This guide will help you set up and run the HER2 clinical question-answering chatbot using Ollama and Python.

## System Requirements

- A machine with AVX-compatible CPU (required by Ollama)
- One of the following operating systems:
  - macOS (Apple Silicon or Intel with Rosetta)
  - Windows (with WSL2)
  - Linux (Ubuntu or similar)

## Step 1: Install Ollama

Ollama runs open-source LLMs locally and must be installed separately.

### macOS

Visit https://ollama.com/download and follow the installation instructions.

### Windows

1. Install WSL2 (https://learn.microsoft.com/en-us/windows/wsl/install)
2. Download and install Ollama from https://ollama.com

### Linux

Follow the official installation steps at https://ollama.com

After installation, verify that Ollama is working by running:

```bash
ollama run mistral
```

This will download and launch the Mistral model for local use.

## Step 2: Clone the Repository and Set Up Python

Create a virtual environment and install Python dependencies:

```bash
git clone <your-repo-url>
cd <your-repo>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

## Step 3: Start Ollama and Launch the Chatbot

In one terminal, start the Ollama model:

```bash
ollama run mistral
```

In a second terminal (inside your virtual environment), run the chatbot:

```bash
python qa_chatbot.py
```

You will see an interactive prompt where you can ask clinical questions related to the HER2 study.

## Notes

- This project is currently configured to work only with locally running Ollama models.
- If you receive an error stating that Ollama is not running, ensure that the model is started with `ollama run mistral` in a separate terminal.
- Ollama must remain running in the background for the chatbot to function.

