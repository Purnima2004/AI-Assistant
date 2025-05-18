# Multilingual AI Assistant

## Description
This project is a Multilingual AI Assistant web application built using Streamlit. It provides multiple assistant functionalities including General AI Assistant, Reading Assistant, Walking Assistant, and Navigation Assistant. The app supports English and Hindi languages and leverages the Groq API for advanced language understanding and zero-shot reasoning.

## Features
- General AI Assistant for answering questions and conversations.
- Reading Assistant to help read text from images, documents, and signs.
- Walking Assistant to provide guidance for safe walking and obstacle detection.
- Navigation Assistant to offer directions, landmark identification, and navigation help.
- Voice, text, and camera input methods.
- Multilingual support for English and Hindi.
- Real-time video processing with OpenCV and WebRTC.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd aiAssistant
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

Open the URL provided by Streamlit in your browser. Choose the desired assistant and interact using text, voice, or camera input.

## Environment Variables

- `GROQ_API_KEY`: API key for accessing the Groq language model.

## Dependencies

- Streamlit
- OpenCV (cv2)
- Pillow (PIL)
- SpeechRecognition
- streamlit-webrtc
- langchain
- langchain-groq
- python-dotenv
- numpy

## License

This project is licensed under the MIT License. See the LICENSE file for details.
