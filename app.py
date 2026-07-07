import pickle
import json
import random

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is available (safe to call every run - no-ops if already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

ERROR_THRESHOLD = 0.25
FALLBACK_RESPONSE = "I'm not sure I understand. Could you rephrase that?"


@st.cache_resource
def load_artifacts():
    """Load the trained model, vocabulary, class labels, and intents once and cache them."""
    model = load_model('artifact/chatbot_model.h5')

    with open('artifact/words.pkl', 'rb') as file:
        words = pickle.load(file)

    with open('artifact/classes.pkl', 'rb') as file:
        classes = pickle.load(file)

    with open('data/intents.json', 'r') as file:
        intents = json.load(file)

    return model, words, classes, intents


model, words, classes, intents = load_artifacts()


def clean_sentence(sentence: str) -> list[str]:
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence: str, words: list[str]) -> np.ndarray:
    """Convert a sentence into a bag-of-words vector matching the trained vocabulary."""
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence: str) -> list[dict]:
    """Predict the intent(s) for a sentence, filtered by confidence threshold."""
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i], "probability": float(r)} for i, r in results]


def get_response(ints: list[dict], intents_json: dict) -> str:
    """Look up a random response for the top predicted intent, with a safe fallback."""
    if not ints:
        return FALLBACK_RESPONSE

    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return FALLBACK_RESPONSE


st.set_page_config(
    page_title="MedIntent-Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 MedIntent-Bot")
st.write("Ask me anything!")
st.caption("⚠️ For informational purposes only. Not a substitute for professional medical advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "debug" in message:
            with st.expander("Debug info"):
                st.write(message["debug"])

prompt = st.chat_input("Type your message...")

if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    intents_pred = predict_class(prompt)
    response = get_response(intents_pred, intents)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "debug": intents_pred}
    )

    with st.chat_message("assistant"):
        st.write(response)
        with st.expander("Debug info"):
            st.write(intents_pred)