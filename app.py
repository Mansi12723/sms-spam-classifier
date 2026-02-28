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
    page_icon="🔐",
    layout="wide"
)

# ---------------- ADVANCED CSS ----------------
st.markdown("""
<style>

/* Animated Background */
body {
    background: linear-gradient(-45deg, #141e30, #243b55, #1c92d2, #f2fcfe);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
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
    font-weight: bold;
    font-size: 20px;
}

/* Hero Section */
.hero-title {
    text-align: center;
    font-size: 65px;
    font-weight: 900;
    color: white;
    animation: fadeIn 2s ease-in;
}

.hero-subtitle {
    text-align: center;
    font-size: 22px;
    color: #f1f1f1;
    margin-bottom: 60px;
    animation: fadeIn 3s ease-in;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(18px);
    padding: 45px;
    border-radius: 25px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    animation: fadeIn 2s ease-in;
}

/* Fade Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(25px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Text Area */
.stTextArea textarea {
    border-radius: 15px;
    border: none;
    padding: 15px;
}

/* Animated Button */
.stButton>button {
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 15px;
    height: 3em;
    width: 100%;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.07);
    box-shadow: 0 0 25px #ff416c;
}

/* Footer */
.footer {
    text-align: center;
    color: white;
    margin-top: 70px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
<div>🔐 SecureMail</div>
<div>Home | Security | Solutions | Contact</div>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown('<div class="hero-title">Enterprise Email Protection</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Advanced Spam Detection & Email Security Platform</div>', unsafe_allow_html=True)

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

    st.subheader("🔎 Email Security Scanner")

    input_text = st.text_area("Paste email content here:", height=200)

    if st.button("🚀 Scan Email"):
        if input_text.strip() == "":
            st.warning("Please enter email content.")
        else:
            with st.spinner("Scanning email..."):
                time.sleep(2)

            transformed = transform_text(input_text)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]
            confidence = round(model.predict_proba(vector_input)[0].max() * 100, 2)

            st.markdown("### Scan Result")

            if result == 1:
                st.error(f"⚠️ Spam Detected (Confidence: {confidence}%)")
            else:
                st.success(f"✅ Email is Safe (Confidence: {confidence}%)")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">© 2026 SecureMail | Enterprise Cybersecurity Solutions</div>', unsafe_allow_html=True)
