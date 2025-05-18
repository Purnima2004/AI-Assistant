import cv2
import threading
import time
import speech_recognition as sr

def listen_for_command():
    """
    Listen for a command from the user.
    
    Returns:
        str: Recognized text or None if recognition failed
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("Listening for commands...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        
        text = recognizer.recognize_google(audio)
        print(f"Command recognized: {text}")
        return text.lower()
    except sr.WaitTimeoutError:
        return None  # No speech detected within the timeout
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return None

def NavigationAssistant():
    """
    Navigation Assistant function that helps users with navigation and directions.
    This function is triggered when the user asks for navigation assistance in English or Hindi.
    Activates the camera and allows speech commands while camera is running.
    """
    print("Navigation Assistant activated!")
    print("I can help you navigate to destinations, provide directions, and find routes.")
    print("How would you like me to assist you with navigation today?")
    print("Say 'exit' or 'quit' to close the Navigation Assistant.")
    
    # Flag to control camera loop
    running = True
    
    # Function to run the camera in a separate thread
    def camera_thread():
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Text to display on the camera feed
        status_text = "Navigation Assistant Active"
        
        while running:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame. Exiting...")
                break
            
            # Add text to the frame
            cv2.putText(frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Blue text for navigation
            
            # Display the frame
            cv2.imshow('Navigation Assistant Camera', frame)
            
            # Break the loop with 'q' key or if running becomes False
            if cv2.waitKey(1) & 0xFF == ord('q') or not running:
                break
        
        # Release everything when done
        cap.release()
        cv2.destroyAllWindows()
    
    # Function to handle voice commands in a separate thread
    def voice_command_thread():
        nonlocal running
        
        while running:
            command = listen_for_command()
            
            if command:
                if any(word in command for word in ["exit", "quit", "close", "stop"]):
                    print("Exiting Navigation Assistant...")
                    running = False
                    break
                elif "help" in command:
                    print("Navigation Assistant Help:")
                    print("- Say 'exit' or 'quit' to close the assistant")
                    print("- Ask about directions, routes, points of interest, etc.")
                else:
                    print(f"Processing command: {command}")
                    # Here you would add specific navigation assistant functionality
                    # based on the user's command
            
            # Brief pause to reduce CPU usage
            time.sleep(0.1)
    
    # Start camera thread
    cam_thread = threading.Thread(target=camera_thread)
    cam_thread.daemon = True
    cam_thread.start()
    
    # Start voice command thread
    voice_thread = threading.Thread(target=voice_command_thread)
    voice_thread.daemon = True
    voice_thread.start()
    
    # Wait for threads to complete (will happen when running becomes False)
    while running:
        time.sleep(0.1)
    
    return "Navigation Assistant closed."