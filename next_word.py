import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model = load_model('next_word_lstm.h5')
with open('tokenizer.pkl','rb') as handle:
    tokenizer = pickle.load(handle)

st.title("Next Word Prediction")
st.write("This is a simple Next Word Prediction App")

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) < max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0) 
    predicted_index = np.argmax(predicted, axis = 1)
    for word, index in tokenizer.word_index.items():
        return word
    return None

input_text = st.text_input('Enter a sentence:', 'The quick brown fox')
if st.button('Predict: Next Word'):
    max_sequences_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequences_len)
    st.write(f"The next word is: {next_word}")

