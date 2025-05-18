import os
import re
import time
import datetime
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading
from dotenv import load_dotenv
import speech_recognition as sr
from pathlib import Path

# Import the assistant functions
from test1 import readingAssistant
from test2 import WalkingAssistant
from test3 import NavigationAssistant

# LangChain imports - using newer syntax
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Load environment variables
load_dotenv()

# Get the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Multilingual AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the beautiful gradient background and UI
def apply_custom_style():
    st.markdown(
        '''
        <style>
        .stApp {
            background: linear-gradient(220.55deg, #61C695 0%, #133114 100%);
            color: white;
        }
        .gradient-header {
            background-image: linear-gradient(90deg, #61C695, #ffffff);
            background-clip: text;
            -webkit-background-clip: text;
            color: transparent;
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
        }
        .assistant-card {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 20px;
            margin: 10px 0;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .assistant-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        }
        .assistant-icon {
            font-size: 40px;
            margin-bottom: 10px;
            text-align: center;
        }
        .assistant-title {
            font-weight: 600;
            margin-bottom: 10px;
            text-align: center;
            color: #61C695;
        }
        .assistant-description {
            font-size: 14px;
            color: #e0e0e0;
        }
        .stButton>button {
            padding: 17px 40px;
            border-radius: 50px;
            cursor: pointer;
            border: 0;
            background-color: #3D9970;
            color: white;
            box-shadow: rgb(0 0 0 / 5%) 0 0 8px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            font-size: 15px;
            transition: all 0.5s ease;
            width: 100%;
        }
        .stButton>button:hover {
            letter-spacing: 3px;
            background-color: #61C695;
            color: white;
            box-shadow: rgb(97 198 149 / 40%) 0px 7px 29px 0px;
        }
        .stButton>button:active {
            letter-spacing: 3px;
            background-color: #2D8A60;
            color: white;
            box-shadow: rgb(97 198 149 / 40%) 0px 0px 0px 0px;
            transform: translateY(10px);
            transition: 100ms;
        }
        .chat-message {
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
        }
        .user-message {
            background-color: rgba(97, 198, 149, 0.3);
            margin-left: 20%;
            border-top-right-radius: 5px;
        }
        .bot-message {
            background-color: rgba(255, 255, 255, 0.1);
            margin-right: 20%;
            border-top-left-radius: 5px;
        }
        .output-area {
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .camera-feed {
            border: 3px solid #3D9970;
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

# Setup language model for zero-shot reasoning using Groq
def setup_language_model():
    """
    Sets up and returns the language model for zero-shot reasoning using Groq.
    
    Returns:
        ChatGroq: Configured language model
    """
    # Initialize the language model with Groq
    if not GROQ_API_KEY:
        st.error("GROQ API key not found. Please enter it in the sidebar.")
        return None
        
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0
    )
    return llm

# Zero-shot task function to determine intent and language
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
    Your task is to determine what type of assistance the user is requesting based on their input.
    
    User input: "{user_text}"
    
    Analyze the input and determine:
    1. If the user is asking for navigation assistance (directions, finding places, etc.)
    2. If the user is asking for reading assistance (reading text, documents, etc.)
    3. If the user is asking for walking assistance (guidance for walking, obstacle detection, etc.)
    4. The language the user is using (English, Hindi, or other)
    
    {format_instructions}
    """
    
    # Create the prompt template
    prompt = PromptTemplate(
        template=prompt_content,
        input_variables=[],
    )
    
    # Get the language model
    llm = setup_language_model()
    if not llm:
        return {
            "assistant_type": "none",
            "confidence": 0,
            "language_detected": "unknown"
        }
    
    # Get the response from the language model
    chain = prompt | llm
    response = chain.invoke({})
    
    try:
        parsed_response = output_parser.parse(response.content)
        return parsed_response
    except Exception as e:
        st.error(f"Error parsing response: {e}")
        return {
            "assistant_type": "none",
            "confidence": 0,
            "language_detected": "unknown"
        }

# Function to handle general knowledge questions
def handle_general_knowledge_question(user_input, language="english"):
    """
    Handle general knowledge questions using the Groq language model
    
    Args:
        user_input (str): User's question
        language (str): Language preference (english or hindi)
        
    Returns:
        str: Response to the question
    """
    try:
        # Setup the language model
        llm = setup_language_model()
        if not llm:
            return "I'm sorry, I need an API key to answer that question."
        
        # Determine if input is in Hindi
        is_hindi = language.lower() == "hindi" or any(char in user_input for char in ["ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ं", "ः", "अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ"])
        
        # Adjust prompt based on language
        if is_hindi:
            system_prompt = "You are a helpful assistant that responds in Hindi when the question is in Hindi and English when the question is in English. Be conversational, helpful, and concise."
        else:
            system_prompt = "You are a helpful assistant. Be conversational, helpful, and concise."
            
        # Generate a response - use a more conversational prompt for general assistant
        response = llm.invoke(f"{system_prompt}\n\nUser: {user_input}")
        return response.content
        
    except Exception as e:
        st.error(f"Error in general knowledge question handling: {e}")
        return "I'm sorry, I couldn't process your question right now."

# Video transformer for camera feed
class VideoTransformer(VideoTransformerBase):
    def __init__(self, assistant_type):
        self.assistant_type = assistant_type
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Store the current frame for processing
        with self.frame_lock:
            self.current_frame = img.copy()
        
        # Apply different processing based on assistant type
        if self.assistant_type == "reading":
            # Add text detection visualization
            img = self.visualize_text_detection(img)
        elif self.assistant_type == "walking":
            # Add obstacle detection visualization
            img = self.visualize_obstacle_detection(img)
        elif self.assistant_type == "navigation":
            # Add navigation guidance visualization
            img = self.visualize_navigation_guidance(img)
            
        return img
    
    def visualize_text_detection(self, img):
        # Placeholder for text detection visualization
        # In a real app, you would use OCR to detect text and highlight it
        cv2.putText(img, "Text Detection Active", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img
    
    def visualize_obstacle_detection(self, img):
        # Placeholder for obstacle detection visualization
        # In a real app, you would use object detection to identify obstacles
        cv2.putText(img, "Obstacle Detection Active", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img
    
    def visualize_navigation_guidance(self, img):
        # Placeholder for navigation guidance visualization
        # In a real app, you would overlay navigation instructions
        cv2.putText(img, "Navigation Guidance Active", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return img
    
    def get_current_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

# Speech recognition function
def recognize_speech():
    """
    Recognizes speech from the microphone and returns the text.
    
    Returns:
        str: Recognized text or error message
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.write("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            
        st.write("Processing speech...")
        # Try to recognize in English first
        try:
            text = recognizer.recognize_google(audio, language="en-US")
            return text
        except:
            # If English fails, try Hindi
            try:
                text = recognizer.recognize_google(audio, language="hi-IN")
                return text
            except:
                return "Sorry, I couldn't understand what you said."
    except Exception as e:
        return f"Error: {str(e)}"

# Main application
def main():
    # Apply custom styling
    apply_custom_style()
    
    # Empty container to reduce top space and add marquee
    st.markdown('<div style="margin-top:-75px;"></div>', unsafe_allow_html=True)
    
    # Add a marquee scrolling text banner
    st.markdown('<marquee behavior="scroll" direction="left" scrollamount="5" style="background: rgba(97, 198, 149, 0.3); padding: 8px; border-radius: 8px; margin-bottom: 15px;">Welcome to AI Assistant Hub! Ask questions, get reading help, walking guidance, and navigation assistance all in one place.</marquee>', unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_input' not in st.session_state:
        st.session_state.last_input = ""
    if 'current_assistant' not in st.session_state:
        st.session_state.current_assistant = None
    if 'video_transformer' not in st.session_state:
        st.session_state.video_transformer = None
    
    # Header
    st.markdown('<h1 class="gradient-header">Multilingual AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.markdown('<p class="assistant-title">Settings</p>', unsafe_allow_html=True)
        
        # API Key input
        api_key = st.text_input("Enter Groq API Key:", value=GROQ_API_KEY if GROQ_API_KEY else "", type="password")
        if api_key and api_key != GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("API Key updated!")
        
        # Language selection
        language = st.selectbox("Preferred Language:", ["English", "Hindi"])
        
        # Reset button
        if st.button("Reset Conversation"):
            st.session_state.chat_history = []
            st.session_state.current_assistant = None
            st.session_state.video_transformer = None
            st.rerun()
    
    # Main content area
    if not st.session_state.current_assistant:
        # Ultra compact header
        st.markdown('<h2 class="gradient-header" style="margin-top:-25px; margin-bottom:8px; font-size:20px;">Choose Your Assistant</h2>', unsafe_allow_html=True)
        
        # Custom CSS for the 3D card design
        st.markdown('''
        <style>
        .parent {
            width: 100%;
            padding: 15px;
            perspective: 1000px;
        }
        
        .card {
            padding-top: 40px;
            border: 3px solid rgba(255, 255, 255, 0.5);
            transform-style: preserve-3d;
            background: linear-gradient(135deg, rgba(0,0,0,0) 18.75%, rgba(19, 49, 20, 0.2) 0 31.25%, rgba(0,0,0,0) 0),
                repeating-linear-gradient(45deg, rgba(19, 49, 20, 0.2) -6.25% 6.25%, rgba(97, 198, 149, 0.1) 0 18.75%);
            background-size: 60px 60px;
            background-position: 0 0, 0 0;
            background-color: rgba(19, 49, 20, 0.6);
            width: 100%;
            min-height: 260px;
            box-shadow: rgba(0, 0, 0, 0.3) 0px 30px 30px -10px;
            transition: all 0.5s ease-in-out;
            margin-bottom: 15px;
            position: relative;
        }
        
        .card:hover {
            background-position: -100px 100px, -100px 100px;
            transform: rotate3d(0.5, 1, 0, 15deg);
        }
        
        .content-box {
            background: rgba(61, 153, 112, 0.85);
            transition: all 0.5s ease-in-out;
            padding: 60px 20px 20px 20px;
            transform-style: preserve-3d;
            height: 100%;
        }
        
        .content-box .card-title {
            display: inline-block;
            color: white;
            font-size: 22px;
            font-weight: 900;
            transition: all 0.5s ease-in-out;
            transform: translate3d(0px, 0px, 30px);
        }
        
        .content-box .card-title:hover {
            transform: translate3d(0px, 0px, 50px);
        }
        
        .content-box .card-content {
            margin-top: 10px;
            font-size: 14px;
            color: #f2f2f2;
            transition: all 0.5s ease-in-out;
            transform: translate3d(0px, 0px, 20px);
            min-height: 70px;
        }
        
        .content-box .card-content:hover {
            transform: translate3d(0px, 0px, 40px);
        }
        
        .icon-box {
            position: absolute;
            top: 20px;
            right: 20px;
            height: 60px;
            width: 60px;
            background: white;
            border: 2px solid rgb(61, 153, 112);
            padding: 10px;
            transform: translate3d(0px, 0px, 50px);
            box-shadow: rgba(0, 0, 0, 0.2) 0px 17px 10px -10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 35px;
            z-index: 100;
        }
        </style>
        ''', unsafe_allow_html=True)
        
        # First row - 2 assistants
        col1, col2 = st.columns(2)
        
        # General AI Assistant
        with col1:
            st.markdown('''
            <div class="parent">
                <div class="card">
                    <div class="icon-box">🤖</div>
                    <div class="content-box">
                        <div class="card-title">AI Assistant</div>
                        <div class="card-content">
                            <marquee behavior="scroll" direction="left" scrollamount="3" onmouseover="this.stop()" onmouseout="this.start()">
                                Ask anything you want! Get answers to general questions, have conversations, or get help with any topic - just like ChatGPT.
                            </marquee>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            if st.button("Select General Assistant", key="general_btn", use_container_width=True):
                st.session_state.current_assistant = "general"
                st.rerun()
            st.markdown('''
                </div>
            </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Reading Assistant
        with col2:
            st.markdown('''
            <div class="parent">
                <div class="card">
                    <div class="icon-box">📖</div>
                    <div class="content-box">
                        <div class="card-title">Reading Assistant</div>
                        <div class="card-content">
                            <marquee behavior="scroll" direction="left" scrollamount="3" onmouseover="this.stop()" onmouseout="this.start()">
                                Helps you read text from images, documents, signs, and more. Perfect for those with visual impairments or reading difficulties.
                            </marquee>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            if st.button("Select Reading Assistant", key="reading_btn", use_container_width=True):
                st.session_state.current_assistant = "reading"
                st.rerun()
            st.markdown('''
                </div>
            </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Second row - 2 assistants
        col1, col2 = st.columns(2)
        
        # Walking Assistant
        with col1:
            st.markdown('''
            <div class="parent">
                <div class="card">
                    <div class="icon-box">🚶</div>
                    <div class="content-box">
                        <div class="card-title">Walking Assistant</div>
                        <div class="card-content">
                            <marquee behavior="scroll" direction="left" scrollamount="3" onmouseover="this.stop()" onmouseout="this.start()">
                                Helps you navigate obstacles and walk safely. Identifies potential hazards and provides real-time guidance.
                            </marquee>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            if st.button("Select Walking Assistant", key="walking_btn", use_container_width=True):
                st.session_state.current_assistant = "walking"
                st.rerun()
            st.markdown('''
                </div>
            </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Navigation Assistant
        with col2:
            st.markdown('''
            <div class="parent">
                <div class="card">
                    <div class="icon-box">🧭</div>
                    <div class="content-box">
                        <div class="card-title">Navigation Assistant</div>
                        <div class="card-content">
                            <marquee behavior="scroll" direction="left" scrollamount="3" onmouseover="this.stop()" onmouseout="this.start()">
                                Helps you find your way around. Provides directions, identifies landmarks, and guides you to your destination.
                            </marquee>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            if st.button("Select Navigation Assistant", key="navigation_btn", use_container_width=True):
                st.session_state.current_assistant = "navigation"
                st.rerun()
            st.markdown('''
                </div>
            </div>
            </div>
            ''', unsafe_allow_html=True)
    
    else:
        # Display the selected assistant interface
        assistant_name = st.session_state.current_assistant.capitalize() + " Assistant"
        st.markdown(f'<p class="assistant-title">{assistant_name}</p>', unsafe_allow_html=True)
        
        # Back button
        if st.button("← Back to Assistant Selection"):
            st.session_state.current_assistant = None
            st.session_state.video_transformer = None
            st.rerun()
        
        # Display chat history
        st.markdown('<div class="output-area">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">Assistant: {message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input methods
        st.markdown('<p class="assistant-title">Input Methods</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Text", "Voice", "Camera"])
        
        with tab1:
            # Text input
            user_input = st.text_area("Type your question or request:", key="text_input")
            if st.button("Send", key="send_text"):
                if user_input:
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Store the user input in session state for reference
                    st.session_state.last_input = user_input
                    
                    # Handle different assistant types
                    if st.session_state.current_assistant == "general":
                        # For general assistant, use the language model to answer any question
                        with st.spinner("Thinking..."):
                            response = handle_general_knowledge_question(user_input, language)
                    elif st.session_state.current_assistant == "reading":
                        st.info(f"Processing your request: {user_input}")
                        response = "I'll help you read that content. Opening Reading Assistant..."
                        # In a real scenario, we'd need to modify readingAssistant() to accept parameters
                    elif st.session_state.current_assistant == "walking":
                        st.info(f"Processing your request: {user_input}")
                        response = "I'll help you with walking guidance. Opening Walking Assistant..."
                        # In a real scenario, we'd need to modify WalkingAssistant() to accept parameters
                    elif st.session_state.current_assistant == "navigation":
                        st.info(f"Processing your request: {user_input}")
                        response = "I'll help you navigate. Opening Navigation Assistant..."
                        # In a real scenario, we'd need to modify NavigationAssistant() to accept parameters
                    else:
                        response = "I'm not sure how to help with that."
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    st.rerun()
        
        with tab2:
            # Voice input
            if st.button("Start Listening", key="start_listening"):
                speech_text = recognize_speech()
                
                if speech_text and speech_text != "Sorry, I couldn't understand what you said." and not speech_text.startswith("Error:"):
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": speech_text})
                    
                    # Store the speech input in session state for reference
                    st.session_state.last_input = speech_text
                    
                    # Handle different assistant types
                    if st.session_state.current_assistant == "general":
                        # For general assistant, use the language model to answer any question
                        with st.spinner("Thinking..."):
                            response = handle_general_knowledge_question(speech_text, language)
                    elif st.session_state.current_assistant == "reading":
                        st.info(f"Processing your voice request: {speech_text}")
                        response = "I'll help you read that content. Opening Reading Assistant..."
                        # In a real scenario, we'd need to modify readingAssistant() to accept parameters
                    elif st.session_state.current_assistant == "walking":
                        st.info(f"Processing your voice request: {speech_text}")
                        response = "I'll help you with walking guidance. Opening Walking Assistant..."
                        # In a real scenario, we'd need to modify WalkingAssistant() to accept parameters
                    elif st.session_state.current_assistant == "navigation":
                        st.info(f"Processing your voice request: {speech_text}")
                        response = "I'll help you navigate. Opening Navigation Assistant..."
                        # In a real scenario, we'd need to modify NavigationAssistant() to accept parameters
                    else:
                        response = "I'm not sure how to help with that."
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    st.rerun()
                else:
                    st.error(speech_text)
        
        with tab3:
            # Camera input
            st.markdown('<div class="camera-feed">', unsafe_allow_html=True)
            ctx = webrtc_streamer(
                key="example",
                video_transformer_factory=lambda: VideoTransformer(st.session_state.current_assistant or "general"),
                async_transform=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if ctx.video_transformer:
                st.session_state.video_transformer = ctx.video_transformer
                
                if st.button("Capture and Process", key="capture_image"):
                    if st.session_state.video_transformer:
                        frame = st.session_state.video_transformer.get_current_frame()
                        
                        if frame is not None:
                            # Process the frame based on the selected assistant
                            if st.session_state.current_assistant == "reading":
                                # For reading assistant, perform OCR
                                # This is a placeholder - in a real app, you would use OCR
                                response = "I can see some text in the image. This is a placeholder for OCR functionality."
                            elif st.session_state.current_assistant == "walking":
                                # For walking assistant, detect obstacles
                                # This is a placeholder - in a real app, you would use object detection
                                response = "I can see the path ahead. This is a placeholder for obstacle detection."
                            elif st.session_state.current_assistant == "navigation":
                                # For navigation assistant, identify landmarks
                                # This is a placeholder - in a real app, you would use landmark recognition
                                response = "I can see where you are. This is a placeholder for navigation guidance."
                            else:
                                response = "I'm not sure how to help with that."
                            
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            
                            st.rerun()
                        else:
                            st.error("No frame captured. Please make sure your camera is working.")
            else:
                st.warning("Camera not available. Please allow camera access and try again.")

if __name__ == "__main__":
    main()