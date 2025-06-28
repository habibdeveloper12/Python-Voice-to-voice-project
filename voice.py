import gradio as gr
import assemblyai as aai
from translate import Translator
from TTS.api import TTS
import uuid
import os

# Set AssemblyAI API Key
# aai.settings.api_key = "2641b438b74a4a4b98e1948cffa0e596"

# Load TTS Model
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

# Transcribe voice to text
def audio_transcription(audio_file):
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
    transcript = aai.Transcriber(config=config).transcribe(audio_file)
    return transcript.text

# Translate text to Bengali, Spanish, Turkish
def lang_translate(text):
    bn = Translator(from_lang="en", to_lang="bn").translate(text)
    es = Translator(from_lang="en", to_lang="es").translate(text)
    tr = Translator(from_lang="en", to_lang="tr").translate(text)
    return bn, es, tr

# Text to speech with Coqui
def text_to_speech(text):
    filename = f"{uuid.uuid4()}.wav"
    tts_model.tts_to_file(text=text, file_path=filename)
    return filename

# Main logic
def real_time_translate(audio_file):
    text = audio_transcription(audio_file)
    bn_text, es_text, tr_text = lang_translate(text)
    bn_audio = text_to_speech(bn_text)
    tr_audio = text_to_speech(tr_text)
    es_audio = text_to_speech(es_text)
    return text, bn_audio, tr_audio, es_audio

# UI
demo = gr.Interface(
    fn=real_time_translate,
    inputs=gr.Audio(source="microphone", type="filepath", label="Speak (short sentence)", max_length=10),
    outputs=[
        gr.Text(label="Original Text"),
        gr.Audio(label="Bengali"),
        gr.Audio(label="Turkish"),
        gr.Audio(label="Spanish")
    ],
    title="Langzila Real-Time Voice Translator",
    description="Speak in English and get near real-time voice responses in Bengali, Turkish, and Spanish."
)

if __name__ == "__main__":
    demo.launch()
