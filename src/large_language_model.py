import text_to_speech_microsoft
import speech_to_text_microsoft
import os
import openai
import time
import keys

done = False
written = True 
client = openai.AzureOpenAI(azure_endpoint=keys.azure_openai_endpoint,
                            api_key=keys.azure_openai_key,
                            api_version=keys.azure_openai_api_version)
discourse = [{"role": "system",
              "content":
              "I am a useful AI assistant. My purpose is to answer questions in a concise, brief manner. Keep responses short and to the point - typically 1-2 sentences maximum unless asked for detailed explanations."}]

def gpt(request):
    discourse.append({"role": "user", "content": request})
    chat = client.chat.completions.create(
        messages = discourse, model = "gpt-4")
    reply = chat.choices[0].message.content
    discourse.append({"role": "assistant", "content": reply})
    return reply

def say(utterance):
    text_to_speech_microsoft.say(utterance)
    if written:
        print(utterance)

def process_utterance(said):
    global done
    if written:
        print(said)
    if "bye" in said:
        done = True
    else:
        response = gpt(said)
        time.sleep(1)
        say(response)
