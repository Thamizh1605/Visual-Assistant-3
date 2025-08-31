# app.py
"""
Main application glue. Starts camera and provides a simple CLI loop for keyboard triggers.
Also starts a small Flask app for future UI (optional).
"""

import threading
import time
import sys
import os
from src.camera import start_camera, capture_image, describe_image
from src.navigation import analyze_frame_for_navigation
from src.voice import speak, start_recording, stop_recording_and_save, transcribe_vosk
from src.qa import ask_question

import cv2

def run_headless_loop(camera_index=0):
    # Start camera
    cap = None
    try:
        cap = start_camera(camera_index)
    except Exception as e:
        print("Camera start failed:", e)
        return

    print("Camera started. Controls:")
    print("  D - Capture & Describe")
    print("  N - Navigation guidance")
    print("  W - Start recording question")
    print("  E - Stop recording & answer question")
    print("  Q - Quit")

    recording_path = None
    latest_image_path = None
    latest_image_desc = "No image captured yet."

    while True:
        # We will read a key from stdin (blocking style). On many terminals, you need to press ENTER.
        # For a nicer cross-platform keypress experience, integrate with pygame or a GUI library.
        key = input("Press (D/N/W/E/Q) and Enter: ").strip().lower()
        if not key:
            continue
        if key == "d":
            try:
                img_path, img_rgb = capture_image(cap)
                latest_image_path = img_path
                desc = describe_image(img_rgb)
                latest_image_desc = desc
                print("Description:", desc)
                speak(desc)
            except Exception as e:
                print("Capture/describe error:", e)
                speak("Sorry, I could not capture the image.")
        elif key == "n":
            try:
                # capture single frame for navigation
                ret, frame = cap.read()
                if not ret:
                    speak("Could not read camera frame for navigation.")
                    continue
                nav = analyze_frame_for_navigation(frame)
                print("Navigation:", nav)
                speak(nav)
            except Exception as e:
                print("Navigation error:", e)
                speak("Navigation failed.")
        elif key == "w":
            try:
                recording_path = start_recording()
                speak("Recording started.")
            except Exception as e:
                print("Recording start failed:", e)
                speak("Cannot start recording.")
        elif key == "e":
            try:
                wav = stop_recording_and_save()
                if not wav:
                    speak("No recording saved.")
                    continue
                speak("Transcribing your question.")
                text = transcribe_vosk(wav)
                if not text:
                    speak("Sorry, transcription failed.")
                    continue
                print("Transcribed:", text)
                speak("You asked: " + text)
                # Ask QA using latest image description
                answer = ask_question(latest_image_desc, text)
                print("Answer:", answer)
                speak(answer)
            except Exception as e:
                print("Stop/transcribe/QA error:", e)
                speak("I had trouble answering your question.")
        elif key == "q":
            speak("Shutting down. Goodbye.")
            break
        else:
            print("Unknown key. Use D/N/W/E/Q.")

    # Cleanup
    try:
        cap.release()
    except Exception:
        pass

if __name__ == "__main__":
    run_headless_loop()
