import streamlit as st
import pickle
import nltk
import string
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="🟣",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Full Purple Background */
.stApp {
    background: linear-gradient(135deg, #2e1065, #4c1d95);
    color: white;
}

/* Hide default Streamlit header/footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Heading Box */
.title-box {
    text-align: center;
    padding: 30px;
    border-radius: 15px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    margin-bottom: 40px;
    border: 1px solid rgba(255,255,255,0.2);
}

.title-text {
    font-size: 48px;
    font-weight: 800;
    color: #e9d5ff;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 200px;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* Text Area */
.stTextArea textarea {
    border-radius: 12px;
    background-color: #1e1e2f;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADING BOX ----------------
st.markdown("""
<div class="title-box">
    <div class="title-text">Email Spam Detection</div>
</div>
""", unsafe_allow_html=True)

# ---------------- NLTK SETUP ----------------
nltk.download('punkt')
nltk.download('stopwords')

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

# ---------------- LOAD MODEL ----------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------- SCAN SECTION ----------------
st.subheader("Scan Email")
input_text = st.text_area("Paste email content below:", height=200)

if st.button("Analyze Email"):
    if input_text.strip() == "":
        st.warning("Please enter email content.")
    else:
        with st.spinner("Scanning..."):
            time.sleep(1.5)

        transformed = transform_text(input_text)
        vector_input = tfidf.transform([transformed])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("⚠ Spam Email Detected")
        else:
            st.success("✔ Legitimate Email")
