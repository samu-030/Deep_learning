import speech_recognition as sr
import pyttsx3

def listen_from_file(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        print("Reading audio file...")
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    print("Virtual Assistant is running...")
    command = listen_from_file("audio.wav") 
    
    if command:
        if "hello" in command:
            speak("Hello! How can I help you?")
        elif "your name" in command:
            speak("I am your virtual assistant.")
        else:
            speak("Sorry, I didn't get that.")
    else:
        print("No command recognized.")

if __name__ == "__main__":
    main()
