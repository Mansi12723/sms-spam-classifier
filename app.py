import streamlit as st
import pickle
import nltk
import string
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SecureMail",
    page_icon="🌈",
    layout="wide"
)

# ---------------- FULL COLOR CSS ----------------
st.markdown("""
<style>

/* Animated Rainbow Background */
body {
    background: linear-gradient(-45deg, #ff0080, #7928ca, #2af598, #009efd);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Navbar */
.navbar {
    padding: 20px 50px;
    display: flex;
    justify-content: space-between;
    color: white;
    font-size: 22px;
    font-weight: bold;
}

/* Hero */
.hero-title {
    text-align: center;
    font-size: 70px;
    font-weight: 900;
    color: #ffffff;
    text-shadow: 0 0 20px #fff;
}

.hero-subtitle {
    text-align: center;
    font-size: 24px;
    color: #f0f0f0;
    margin-bottom: 60px;
}

/* Colorful Glass Card */
.card {
    background: linear-gradient(135deg, rgba(255,255,255,0.25), rgba(255,255,255,0.15));
    backdrop-filter: blur(20px);
    padding: 50px;
    border-radius: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    border: 2px solid rgba(255,255,255,0.3);
}

/* Text Area */
.stTextArea textarea {
    border-radius: 15px;
    padding: 15px;
    border: none;
}

/* Neon Button */
.stButton>button {
    background: linear-gradient(90deg, #00f2fe, #4facfe);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 20px;
    height: 3.5em;
    width: 100%;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 30px #00f2fe;
}

/* Footer */
.footer {
    text-align: center;
    color: white;
    margin-top: 80px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
<div>🌈 SecureMail</div>
<div>Home | Features | Solutions | Contact</div>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown('<div class="hero-title">Colorful Email Protection</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Powerful Spam Detection Platform</div>', unsafe_allow_html=True)

# ---------------- NLTK ----------------
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

# ---------------- CENTER SECTION ----------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🔍 Scan Your Email")

    input_text = st.text_area("Paste your email content:", height=220)

    if st.button("🚀 Scan Now"):
        if input_text.strip() == "":
            st.warning("Please enter email content.")
        else:
            with st.spinner("Scanning..."):
                time.sleep(2)

            transformed = transform_text(input_text)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]
            confidence = round(model.predict_proba(vector_input)[0].max() * 100, 2)

            st.markdown("### 🎯 Result")

            if result == 1:
                st.error(f"🚨 Spam Detected | Confidence: {confidence}%")
            else:
                st.success(f"✅ Safe Email | Confidence: {confidence}%")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">© 2026 SecureMail | Modern Cybersecurity Platform</div>', unsafe_allow_html=True)
