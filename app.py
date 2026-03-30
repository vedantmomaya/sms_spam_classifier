

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# --- Streamlit page config ---
st.set_page_config(
    page_title="SMS/Email Spam Classifier",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for beautification ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f7fa;
    }
    .stTextArea textarea {
        background-color: #fffbe7;
        font-size: 1.1em;
        border-radius: 8px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4e54c8 0%, #8f94fb 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
        margin-top: 1em;
    }
    .stHeader, .stTitle {
        color: #4e54c8;
    }
    .result-box {
        background: #e0e7ff;
        border-radius: 10px;
        padding: 1.5em;
        margin-top: 1.5em;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        color: #22223b;
        box-shadow: 0 2px 8px rgba(78,84,200,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/561/561127.png",
    width=80
)
st.sidebar.title("About")
st.sidebar.info(
    "This app uses Natural Language Processing and Machine Learning to classify SMS and Email messages as Spam or Not Spam.\n\nBuilt with Streamlit, NLTK, and scikit-learn."
)

# Ensure NLTK punkt, punkt_tab, and stopwords data are available
for resource, path in [
    ('punkt', 'tokenizers/punkt'),
    ('punkt_tab', 'tokenizers/punkt_tab'),
    ('stopwords', 'corpora/stopwords')
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.markdown("""
<h1 style='text-align: center; color: #4e54c8; margin-bottom: 0.5em;'>📩 SMS/Email Spam Classifier</h1>
<p style='text-align: center; color: #22223b; font-size: 1.1em;'>
Paste your message below and click <b>Predict</b> to check if it's spam or not.
</p>
""", unsafe_allow_html=True)

input_sms = st.text_area("Enter your message here:", height=150)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button('Predict')

if predict_btn and input_sms.strip():
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown('<div class="result-box">🚫 <span style="color:#c1121f">Spam</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box">✅ <span style="color:#2d6a4f">Not Spam</span></div>', unsafe_allow_html=True)
