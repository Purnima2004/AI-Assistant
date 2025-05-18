# AI Assistant with Robust Speech Recognition Functionality

A voice-controlled AI assistant with multilingual support (English and Hindi) that can direct users to different specialized assistant functions based on their requests. The assistant can handle reading, walking, and navigation tasks, as well as answer general knowledge questions.

## Features

- **Multilingual Support**: Understands and responds to commands in both English and Hindi
- **Speech Recognition**: Converts speech to text using Google's Speech Recognition API
- **Text-to-Speech**: Provides voice responses using pyttsx3
- **Intent Detection**: Uses LangChain with Groq API for advanced zero-shot reasoning
- **Fallback Mechanism**: Implements keyword detection as a backup
- **Camera Integration**: Uses OpenCV for visual assistance features
- **Specialized Assistants**:
  - Reading Assistant (helps with reading text and documents)
  - Walking Assistant (helps with navigation on foot)
  - Navigation Assistant (helps with directions and routes)
- **General Knowledge**: Answers general questions using LLM

## UI Preview

Here's how the AI Assistant interface looks like:

![image](https://github.com/user-attachments/assets/3ff7b9db-423d-4697-a99f-5b90b84b408e)

*Screenshot: The main interface of the AI Assistant showing the chat interface and assistant selection*

*Note: Replace the above comment with your actual screenshot path once you have it.*

## Prerequisites

Before you can use this AI assistant, you need to install the following dependencies:

### Hardware Requirements

- Microphone (for speech input)
- Camera (for visual assistance features)
- Speakers (for text-to-speech output)

### Software Requirements

- Python 3.8 or higher

### API Keys

- **Groq API Key**: Used for the language model. Get yours at [https://console.groq.com/](https://console.groq.com/)

## Installation

1. Clone this repository or download the code files.

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Required Python Packages

The following packages need to be installed:

- **pyttsx3**: Text-to-speech conversion library
  ```bash
  pip install pyttsx3
  ```

- **python-dotenv**: For loading environment variables
  ```bash
  pip install python-dotenv
  ```

- **SpeechRecognition**: For speech recognition capabilities
  ```bash
  pip install SpeechRecognition
  ```

- **sounddevice**: For capturing audio from microphone
  ```bash
  pip install sounddevice
  ```

- **NumPy**: For numerical operations
  ```bash
  pip install numpy
  ```

- **LangChain**: For AI chain operations
  ```bash
  pip install langchain
  ```

- **LangChain Groq**: For integration with Groq API
  ```bash
  pip install langchain-groq
  ```

- **OpenCV**: For camera operations
  ```bash
  pip install opencv-python
  ```

- **PyAudio**: Required by SpeechRecognition for microphone access
  ```bash
  pip install PyAudio
  ```

Alternatively, you can install all requirements at once using:

```bash
pip install pyttsx3 python-dotenv SpeechRecognition sounddevice numpy langchain langchain-groq opencv-python PyAudio
```

Or use the included requirements.txt file:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your microphone and camera are connected and working.

2. Run the main script:
   ```bash
   python main.py
   ```

3. The assistant will start with a greeting and will listen for your commands.

4. Speak your request in English or Hindi. For example:
   - "I need help with reading this document" (activates Reading Assistant)
   - "मुझे चलने में मदद चाहिए" (activates Walking Assistant)
   - "Show me directions to the nearest coffee shop" (activates Navigation Assistant)
   - "What is the capital of France?" (triggers general knowledge question)

5. Say "exit", "quit", or "bye" to exit the assistant.

## Troubleshooting

- **Speech Recognition Issues**: Make sure your microphone is properly connected and configured. Check that you have a working internet connection for Google's Speech Recognition.
- **Text-to-Speech Issues**: Ensure your speakers are working and the volume is turned up.
- **Camera Issues**: Verify that your camera is properly connected and not being used by another application.
- **API Key Errors**: Confirm that your Groq API key is correctly set in the `.env` file.



## Acknowledgments

- Google Speech Recognition for speech-to-text capabilities
- Groq API for language model services
