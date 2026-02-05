import text_to_speech_microsoft
import speech_to_text_microsoft
import large_language_model
import os
import openai
import time
import keys

# microphone imports
import sounddevice as sd
import numpy as np
import wave
import tempfile
import signal
import webrtcvad
import azure.cognitiveservices.speech as speechsdk
import sys

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'

# Graceful exit on Ctrl+C
def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


signal.signal(signal.SIGINT, signal_handler)


def record_audio_vad(filename, aggressiveness=2, max_record_time=30, silence_timeout=1.0):
    """Record audio using Voice Activity Detection to automatically start/stop recording"""
    print("Listening... (speak to start, stop talking to end)")
    vad = webrtcvad.Vad(aggressiveness)

    sample_rate = SAMPLE_RATE
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000)
    buffer = []
    silence_start = None
    started = False
    start_time = time.time()

    def is_speech(frame_bytes):
        return vad.is_speech(frame_bytes, sample_rate)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        while True:
            frame, _ = stream.read(frame_size)
            frame_bytes = frame.tobytes()
            if is_speech(frame_bytes):
                buffer.append(frame)
                if not started:
                    started = True
                    print("Speech detected, recording...")
                silence_start = None
            else:
                if started:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_timeout:
                        print("Silence detected, stopping recording.")
                        break
            if started and (time.time() - start_time > max_record_time):
                print("Max record time reached, stopping.")
                break
    
    if buffer:
        audio = np.concatenate(buffer, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(np.dtype('int16').itemsize)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        print(f"Saved recording to {filename}")
    else:
        print("No speech detected.")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(np.dtype('int16').itemsize)
            wf.setframerate(sample_rate)
            wf.writeframes(b"")


def transcribe_audio_file(audio_path):
    """Transcribe audio file using Microsoft Speech Service"""
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=keys.azure_key,
            region=keys.azure_region)
        audio_input = speechsdk.AudioConfig(filename=audio_path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_input)
        
        result = speech_recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
            return ""
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
            return ""
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""


def main():
    # Set up text-to-speech service
    text_to_speech_microsoft.start()
    time.sleep(1)

    print("Starting conversation...")
    text_to_speech_microsoft.say("How do you do. Please tell me your problem.")
    
    # Wait for initial greeting to finish
    while len(text_to_speech_microsoft.things_to_say) > 0 or text_to_speech_microsoft.is_speaking:
        time.sleep(0.1)
    
    while True:
        print("\n--- New Interaction ---")
        
        # Wait for TTS to finish before recording (efficient flag checking)
        while len(text_to_speech_microsoft.things_to_say) > 0 or text_to_speech_microsoft.is_speaking:
            time.sleep(0.1)
        
        print("Ready to listen...")
        # Record audio using VAD
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name
        
        record_audio_vad(audio_path, aggressiveness=1, silence_timeout=3.0, max_record_time=45)
        
        # Transcribe the recorded audio
        text = transcribe_audio_file(audio_path)
        os.remove(audio_path)
        
        if not text.strip():
            print("No speech detected. Try again.")
            continue
        
        print(f"You said: {text}")
        
        # Check for exit conditions
        if any(word in text.lower() for word in ["bye", "goodbye", "exit", "quit"]):
            text_to_speech_microsoft.say("Goodbye!")
            break
        
        # Process with LLM
        response = large_language_model.process_utterance(text)
        if response:
            print(f"Response: {response}")
            print("Speaking response...")
            text_to_speech_microsoft.say(response)
        
        # Check if conversation should end
        if large_language_model.done:
            break

    # Wait for final TTS to complete
    while len(text_to_speech_microsoft.things_to_say) > 0 or text_to_speech_microsoft.is_speaking:
        time.sleep(0.1)
    
    text_to_speech_microsoft.stop()


if __name__ == "__main__":
    main()

