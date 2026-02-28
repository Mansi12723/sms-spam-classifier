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

# ---------------- FULL PAGE BACKGROUND FIX ----------------
st.markdown("""
<style>

/* Apply background to full Streamlit app */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Remove default Streamlit header & footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Navbar */
.navbar {
    background: #1e293b;
    padding: 18px 60px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 8px;
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 60px 20px;
}

.hero-title {
    font-size: 55px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 20px;
    color: #cbd5e1;
    margin-top: 10px;
}

/* Card */
.card {
    background: #1e293b;
    padding: 45px;
    border-radius: 18px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    border: 1px solid #334155;
}

/* Text Area */
.stTextArea textarea {
    border-radius: 12px;
    padding: 15px;
    background-color: #0f172a;
    color: white;
    border: 1px solid #334155;
}

/* Button */
.stButton>button {
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 80px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
<div>🔐 SecureMail</div>
<div>Home | Solutions | Enterprise | Contact</div>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
<div class="hero-title">Enterprise Email Security</div>
<div class="hero-subtitle">Advanced spam detection and protection system</div>
</div>
""", unsafe_allow_html=True)

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

# ---------------- CENTER CONTENT ----------------
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
                st.error(f"Spam Detected (Confidence: {confidence}%)")
            else:
                st.success(f"Email is Safe (Confidence: {confidence}%)")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© 2026 SecureMail Technologies | Privacy Policy | Terms
</div>
""", unsafe_allow_html=True)
