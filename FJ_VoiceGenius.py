import streamlit as st
import speech_recognition as sr
from google.generativeai import GenerativeModel
from googletrans import Translator, LANGUAGES
from textblob import TextBlob
from gtts import gTTS
import os
import uuid
import spacy
from transformers import pipeline
import plotly.graph_objs as go


# Define TextGenerator class
class TextGenerator:
    def __init__(self):
        self.api_key = "AIzaSyDx7Vykf8bDqi5tLZlsr0HKi0xEKnnhzL4"  # Replace with your API key

    def generate_content(self, prompt):
        try:
            model = GenerativeModel(model_name="gemini-pro", api_key=self.api_key)
            # Configuration for text generation
            generation_config = {
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            model.configure(api_key=self.api_key)
            model.set_generation_config(generation_config)
            model.set_safety_settings(safety_settings)

            # Generate text based on prompt
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"An error occurred during text generation: {str(e)}")
            return None

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening... Please speak clearly into the microphone.")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success("Speech recognition successful.")
        st.write(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, the audio could not be understood.")
        return None
    except sr.RequestError as e:
        st.error(f"Error fetching speech recognition results: {e}")
        return None

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    try:
        translated_text = translator.translate(text, dest=target_language).text
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return None

# Function to play audio
def play_audio(text, target_language):
    tts = gTTS(text=text, lang=target_language)
    filename = str(uuid.uuid4()) + ".mp3"  # Unique filename to avoid clashes
    try:
        tts.save(filename)
        os.system(filename)
    except Exception as e:
        st.error(f"Error during audio playback: {e}")
    finally:
        # Attempt to remove the temporary audio file, handling potential errors
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            st.error(f"Error deleting temporary audio file: {e}")

# Function for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive", sentiment_score
    elif sentiment_score < 0:
        return "Negative", sentiment_score
    else:
        return "Neutral", sentiment_score

# Function for keyword extraction
def extract_keywords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # Extract nouns and proper nouns as keywords
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return keywords

# Function for text summarization
def summarize_text(text):
    summarization_pipeline = pipeline("summarization")
    summarized_text = summarization_pipeline(text, max_length=150, min_length=30, do_sample=False)
    return summarized_text[0]['summary_text']

def main():
    st.title("FJ VoiceGenius")
    st.markdown(
        """
        <style>
        .reportview-container {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write("Welcome to FJ VoiceGenius, your ultimate companion for speech recognition and text generation! FJ VoiceGenius is a cutting-edge application designed to transform spoken words into written text with unparalleled accuracy and efficiency. Whether you're jotting down notes, brainstorming ideas, or crafting content, FJ VoiceGenius empowers you to harness the power of your voice like never before")

    st.subheader("Instructions:")
    st.write("- Click the 'Start Recording' button to begin.")
    st.write("- Speak clearly into the microphone during recording.")
    st.write("- Click the button again to stop recording, view the recognized text, and generate content based on it.")

    target_language = st.selectbox("Select Target Language", LANGUAGES.values())  

    if target_language != "Select Language":
        if st.button("Start Recording"):
            recognized_text = recognize_speech()
            if recognized_text:
                st.text_area("Recognized Text", recognized_text, key="recognized_text_area")
                # Perform sentiment analysis
                sentiment, sentiment_score = analyze_sentiment(recognized_text)
                st.write(f"Sentiment: {sentiment}")
                # Visualize sentiment with bars
                sentiment_bars = go.Figure(go.Bar(
                    x=["Sentiment"],
                    y=[sentiment_score],
                    text=[f"{sentiment} ({sentiment_score:.2f})"],
                    textposition='auto',
                    marker=dict(color=["green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "gray"]),
                    name="Sentiment"
                ))
                sentiment_bars.update_layout(
                    title="Sentiment Analysis",
                    xaxis_title="Sentiment",
                    yaxis_title="Sentiment Score",
                    legend=dict(
                        title="Sentiment",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    hovermode="x",
                    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                )
                st.plotly_chart(sentiment_bars)

                # Extract keywords (if needed)
                keywords = extract_keywords(recognized_text)
                st.write("Keywords:")
                st.write(keywords)

                # Summarize text
                summarized_text = summarize_text(recognized_text)
                st.write("Summary:")
                st.write(summarized_text)

                if st.button("Generate Text"):
                    text_generator = TextGenerator()
                    with st.spinner("Generating text..."):
                        generated_text = text_generator.generate_content(recognized_text)
                    if generated_text:
                        st.text_area("Generated Text", generated_text, key="generated_text_area")
                        st.write("Generated Text:")
                        st.write(generated_text)
                if st.button("Translate"):
                    translated_text = translate_text(recognized_text, target_language)
                    st.text_area("Translated Text", translated_text, key="translated_text_area")
                    st.write("Translated Text:")
                    st.write(translated_text)
                if st.button("Play Audio"):
                    play_audio(recognized_text, target_language)

if __name__ == "__main__":
    main()
