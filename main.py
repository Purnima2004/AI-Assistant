import os
import re
import time
import datetime
import getpass
import pyttsx3
from dotenv import load_dotenv
import speech_recognition as sr
import sounddevice as sd
import numpy as np

# LangChain imports - using newer syntax
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Import the assistant functions
from test1 import readingAssistant
from test2 import WalkingAssistant
from test3 import NavigationAssistant

# Load environment variables
load_dotenv()

# Get the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("GROQ_API_KEY not found in environment variables.")
    GROQ_API_KEY = getpass.getpass("Enter your Groq API key: ")
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize text-to-speech engine
def initialize_tts_engine():
    """Initialize and configure the text-to-speech engine"""
    engine = pyttsx3.init()
    # Set properties
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    
    # Get available voices
    voices = engine.getProperty('voices')
    # Set a voice - typically index 1 is female voice on Windows
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)  # Use female voice if available
    
    return engine

# Text-to-speech function
def speak(text, engine):
    """Speak the provided text"""
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

# Sets up and returns the language model for zero-shot reasoning using Groq.
def setup_language_model():
    """
    Sets up and returns the language model for zero-shot reasoning using Groq.
    
    Returns:
        ChatGroq: Configured language model
    """
    # Initialize the language model with Groq using newer syntax
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",  # You can also use "mixtral-8x7b-32768" or other Groq models
        temperature=0
    )
    return llm

def zero_shot_task(user_text):
    """
    Advanced zero-shot reasoning to understand user intent using LangChain with Groq.
    Detects if user is asking for navigation, reading, or walking assistance in English or Hindi.
    
    Args:
        user_text (str): User's input text
        
    Returns:
        dict: Structured output with detected intent and confidence
    """
    if not user_text:
        return {
            "assistant_type": "none",
            "confidence": 0,
            "language_detected": "unknown"
        }
        
    # Define output schemas
    response_schemas = [
        ResponseSchema(
            name="assistant_type",
            description="The type of assistant the user is requesting: 'navigation', 'reading', 'walking', or 'none'",
        ),
        ResponseSchema(
            name="confidence",
            description="Confidence level (0-100) that the detected assistant type is correct",
        ),
        ResponseSchema(
            name="language_detected",
            description="The language detected in the user query: 'english', 'hindi', or 'other'",
        )
    ]
    
    # Create output parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # Create the prompt content
    prompt_content = f"""
    You are an AI assistant that understands both English and Hindi languages.
    Your task is to determine what type of assistance the user is requesting.
    
    The user might be asking for:
    1. Navigation assistance (navigation panel, directions, maps, routes, etc.)
        - English examples: "open navigation", "show me the way", "I need directions", "navigation panel", etc.
        - Hindi examples: "नेविगेशन खोलो", "रास्ता दिखाओ", "मुझे दिशा-निर्देश चाहिए", "नेविगेशन पैनल", etc.
    
    2. Reading assistance (reading help, document reading, etc.)
        - English examples: "reading assistant", "help me read this", "read this document", etc.
        - Hindi examples: "पढ़ने में मदद करो", "यह दस्तावेज़ पढ़ें", "रीडिंग असिस्टेंट", etc.
    
    3. Walking assistance (walking help, guidance while walking, etc.)
        - English examples: "walking assistant", "help me walk", "guide my steps", etc.
        - Hindi examples: "चलने में मदद करो", "वॉकिंग असिस्टेंट", "मेरे कदमों का मार्गदर्शन करें", etc.
    
    User input: {user_text}
    
    {format_instructions}
    """
    
    try:
        # Get the language model
        llm = setup_language_model()
        
        # Use the newer invoke method directly
        result = llm.invoke(prompt_content)
        
        # Parse the result - extract content from the result
        content = result.content if hasattr(result, 'content') else str(result)
        parsed_result = output_parser.parse(content)
        
        # Ensure confidence is an integer
        if isinstance(parsed_result["confidence"], str):
            parsed_result["confidence"] = int(parsed_result["confidence"])
            
        return parsed_result
    except Exception as e:
        print(f"Error with LangChain processing: {e}")
        # Fallback to simple keyword matching
        return fallback_intent_detection(user_text)

def fallback_intent_detection(user_text):
    """
    Fallback method for intent detection using simple keyword matching.
    Used if the LangChain parsing fails.
    
    Args:
        user_text (str): User's input text
        
    Returns:
        dict: Detected intent and confidence
    """
    user_text = user_text.lower()
    
    # Navigation keywords (English and Hindi)
    navigation_keywords = [
        "navigation", "navigate", "directions", "maps", "routes", "way", "path",
        "नेविगेशन", "दिशा", "मार्ग", "रास्ता", "नक्शा", "नेविगेट"
    ]
    
    # Reading keywords (English and Hindi)
    reading_keywords = [
        "reading", "read", "document", "text", "book",
        "पढ़ना", "पढ़", "दस्तावेज़", "किताब", "पाठ", "रीडिंग"
    ]
    
    # Walking keywords (English and Hindi)
    walking_keywords = [
        "walking", "walk", "step", "move", "foot",
        "चलना", "कदम", "पैदल", "चल", "वॉकिंग"
    ]
    
    # Count keyword matches
    navigation_count = sum(1 for word in navigation_keywords if word in user_text)
    reading_count = sum(1 for word in reading_keywords if word in user_text)
    walking_count = sum(1 for word in walking_keywords if word in user_text)
    
    # Determine the most likely intent
    max_count = max(navigation_count, reading_count, walking_count)
    
    if max_count == 0:
        return {
            "assistant_type": "none",
            "confidence": 0,
            "language_detected": "unknown"
        }
    
    # Detect language (simple approach)
    english_pattern = r'[a-zA-Z]'
    hindi_pattern = r'[\u0900-\u097F]'
    
    english_chars = len(re.findall(english_pattern, user_text))
    hindi_chars = len(re.findall(hindi_pattern, user_text))
    
    language = "english" if english_chars > hindi_chars else "hindi"
    
    # Determine intent based on keyword count
    if max_count == navigation_count:
        return {
            "assistant_type": "navigation",
            "confidence": min(navigation_count * 20, 100),
            "language_detected": language
        }
    elif max_count == reading_count:
        return {
            "assistant_type": "reading",
            "confidence": min(reading_count * 20, 100),
            "language_detected": language
        }
    else:
        return {
            "assistant_type": "walking",
            "confidence": min(walking_count * 20, 100),
            "language_detected": language
        }

def generate_greeting():
    """Generate a time-appropriate greeting message"""
    current_hour = datetime.datetime.now().hour
    
    if 5 <= current_hour < 12:
        greeting = "Good morning"
    elif 12 <= current_hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
    
    greeting += ", you are doing wonderful today. How can I help you to make your day better?"
    return greeting

def handle_general_question(user_input, tts_engine):
    """
    Handle general conversational questions that are not related to the specific assistants
    
    Args:
        user_input (str): User's input text
        tts_engine: Text-to-speech engine
        
    Returns:
        bool: True if handled, False if not a general question
    """
    user_input_lower = user_input.lower()
    
    # Who are you questions
    if any(phrase in user_input_lower for phrase in ["who are you", "what are you", "your name", "introduce yourself"]):
        response = "I am your AI Assistant. I will help you solve all your problems"
        speak(response, tts_engine)
        return True
        
    # How are you questions    
    elif any(phrase in user_input_lower for phrase in ["how are you", "how do you do", "how's it going"]):
        response = "I'm functioning well, thank you for asking! How can I assist you today?"
        speak(response, tts_engine)
        return True
        
    # What can you do questions
    elif any(phrase in user_input_lower for phrase in ["what can you do", "your capabilities", "help me with", "what do you do"]):
        response = "I can assist you with anything you want. Just ask me I will be there for youright away."
        speak(response, tts_engine)
        return True
        
    # Time questions
    elif any(phrase in user_input_lower for phrase in ["what time", "current time", "time now"]):
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        response = f"The current time is {current_time}"
        speak(response, tts_engine)
        return True
        
    # Date questions
    elif any(phrase in user_input_lower for phrase in ["what date", "what day", "today's date"]):
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        response = f"Today is {current_date}"
        speak(response, tts_engine)
        return True
    
    # Not a general question we recognize
    return False

def handle_general_knowledge_question(user_input, tts_engine):
    """
    Handle general knowledge questions using the Groq language model
    
    Args:
        user_input (str): User's question
        tts_engine: Text-to-speech engine
        
    Returns:
        bool: True if the question was handled
    """
    try:
        # Initialize the language model
        llm = setup_language_model()
        
        # Create a prompt for general knowledge questions
        prompt = f"""
        
        
        User question: {user_input}
        
        
        """
        
        # Get the response
        result = llm.invoke(prompt)
        answer = result.content if hasattr(result, 'content') else str(result)
        
        # Clean up the answer if needed
        answer = answer.strip()
        
        # Speak the answer
        speak(answer, tts_engine)
        return True
        
    except Exception as e:
        print(f"Error handling general knowledge question: {e}")
        return False

def process_user_input(user_input, tts_engine):
    """
    Process user input and route to the appropriate assistant.
    
    Args:
        user_input (str): User's input text
        tts_engine: Text-to-speech engine
        
    Returns:
        str: Response from the appropriate assistant
    """
    # First check if it's a general question about the assistant
    if handle_general_question(user_input, tts_engine):
        return None
    
    # Use zero-shot reasoning to determine the user's intent
    intent_result = zero_shot_task(user_input)
    
    print(f"Detected intent: {intent_result}")
    
    # Decision threshold - only trigger assistants if confidence is above this value
    confidence_threshold = 50
    
    if intent_result["confidence"] >= confidence_threshold:
        assistant_type = intent_result["assistant_type"]
        
        if assistant_type == "navigation":
            speak(f"Opening the Navigation Assistant for you.", tts_engine)
            return NavigationAssistant()
        elif assistant_type == "reading":
            speak(f"Opening the Reading Assistant for you.", tts_engine)
            return readingAssistant()
        elif assistant_type == "walking":
            speak(f"Opening the Walking Assistant for you.", tts_engine)
            return WalkingAssistant()
    
    # If not a request for a specific assistant, treat it as a general knowledge question
    if handle_general_knowledge_question(user_input, tts_engine):
        return None
    
    # If we got here, we couldn't handle the request
    response = "I'm not sure which assistant you need. Please try again"
    speak(response, tts_engine)
    return response

def listen_for_speech():
    """
    Listen for user speech and convert it to text.
    
    Returns:
        str: Transcribed text from speech or None if recognition failed
    """
    recognizer = sr.Recognizer()
    sample_rate = 16000
    duration = 5  # Record for 5 seconds
    
    print("Listening... (speak now)")
    
    try:
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1)
        sd.wait()
        
        # Convert audio numpy array to audio data format for recognizer
        audio_data = (audio_data * 32767).astype(np.int16)
        audio = sr.AudioData(audio_data.tobytes(), sample_rate, 2)
        
        # Recognize speech using Google's Speech Recognition
        text = recognizer.recognize_google(audio)  # Will auto-detect language
        print(f"You said: {text}")
        return text
    
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from speech service; {e}")
        return None
    except Exception as e:
        print(f"Error during speech recognition: {e}")
        return None

def main():
    """
    Main function that runs the AI assistant.
    """
    try:
        # Initialize text-to-speech engine
        tts_engine = initialize_tts_engine()
        voice_available = True
    except Exception as e:
        print(f"Error initializing text-to-speech engine: {e}")
        voice_available = False
    
    print("AI Assistant started. Supports English and Hindi.")
    
    # Check if voice is available
    if not voice_available:
        print("Error: Text-to-speech functionality is not available.")
        print("Please check your system's audio configuration and try again.")
        return  # Exit the function, which will end the program
    
    # Welcome the user with a greeting
    greeting = generate_greeting()
    speak(greeting, tts_engine)
    
    while True:
        print("\nListening for your command...")
        
        # Get speech input
        user_input = listen_for_speech()
        
        # If speech recognition failed, try again
        if not user_input:
            speak("I didn't catch that. Could you please try again?", tts_engine)
            continue
            
        # Check for exit command - ensure user_input is a string
        if isinstance(user_input, str) and (user_input.lower() in ["exit", "quit", "bye"] or "thank you for your help" in user_input.lower()):
            speak("Exiting AI Assistant. Goodbye and have a wonderful day!", tts_engine)
            break
        
        # Process the user input and get response - ensure user_input is a string
        if isinstance(user_input, str):
            print("Processing your request...")
            response = process_user_input(user_input, tts_engine)
            
            # If response is a string (not handled by a general question handler), speak it
            if response and isinstance(response, str):
                speak(response, tts_engine)
        else:
            speak("I encountered an issue processing your request. Let's try again.", tts_engine)

if __name__ == "__main__":
    # Run the main function
    main()