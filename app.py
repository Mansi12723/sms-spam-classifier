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
    layout="wide"
)

# ---------------- PURPLE THEME CSS ----------------
st.markdown("""
<style>

/* Full Background */
.stApp {
    background: linear-gradient(135deg, #1a0b2e, #2e1065, #4c1d95);
    color: white;
}

/* Hide Streamlit default header/footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Navbar */
.navbar {
    background: rgba(255,255,255,0.08);
    padding: 18px 60px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 10px;
    backdrop-filter: blur(10px);
    font-size: 18px;
    font-weight: 600;
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 80px 20px 30px 20px;
}

.hero-title {
    font-size: 56px;
    font-weight: 900;
    background: linear-gradient(90deg, #c084fc, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 20px;
    color: #ddd6fe;
    margin-top: 10px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.08);
    padding: 45px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.15);
}

/* Text Area */
.stTextArea textarea {
    border-radius: 12px;
    padding: 15px;
    background-color: rgba(0,0,0,0.3);
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #c084fc;
}

.footer {
    text-align: center;
    color: #c4b5fd;
    margin-top: 80px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
<div>🟣 SecureMail</div>
<div>Home | Solutions | Contact</div>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <div class="hero-title">Email Spam Detection</div>
    <div class="hero-subtitle">
        This system intelligently analyzes email content and automatically classifies messages as spam or legitimate.
    </div>
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

# ---------------- CENTER CARD ----------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Scan Email")
    input_text = st.text_area("Paste email content below:", height=200)

    if st.button("Analyze Email"):
        if input_text.strip() == "":
            st.warning("Please enter email content.")
        else:
            with st.spinner("Scanning email..."):
                time.sleep(1.5)

            transformed = transform_text(input_text)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]
            confidence = round(model.predict_proba(vector_input)[0].max() * 100, 2)

            st.markdown("### Result")

            if result == 1:
                st.error(f"⚠ Spam Email Detected ({confidence}% confidence)")
            else:
                st.success(f"✔ Legitimate Email ({confidence}% confidence)")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© 2026 SecureMail | Email Protection System
</div>
""", unsafe_allow_html=True)
