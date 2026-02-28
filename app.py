import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SecureMail AI",
    page_icon="🚀",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Navbar */
.navbar {
    background-color: rgba(255,255,255,0.08);
    padding: 15px 40px;
    border-radius: 10px;
    margin-bottom: 30px;
    display: flex;
    justify-content: space-between;
    color: white;
    font-weight: 600;
}

/* Hero Section */
.hero-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    text-align: center;
    font-size: 20px;
    color: #d1d5db;
    margin-bottom: 40px;
}

/* Card */
.card {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
}

/* Textarea */
.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #ddd;
}

/* Footer */
.footer {
    text-align: center;
    color: #d1d5db;
    margin-top: 50px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
<div>🚀 SecureMail AI</div>
<div>Home | Services | About | Contact</div>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown('<p class="hero-title">Enterprise Email Security</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Spam Detection for Modern Businesses</p>', unsafe_allow_html=True)

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

# ---------------- MAIN SECTION ----------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Analyze Your Email")

    input_text = st.text_area("Paste email content below:", height=200)

    if st.button("🔎 Run AI Scan"):
        if input_text.strip() == "":
            st.warning("Please enter email content.")
        else:
            transformed = transform_text(input_text)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]

            st.write("")
            st.markdown("### Scan Result")

            if result == 1:
                st.error("⚠️ Threat Detected: This email is SPAM.")
            else:
                st.success("✅ Secure: This email is safe.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">© 2026 SecureMail AI Technologies | Privacy Policy | Terms of Service</div>', unsafe_allow_html=True)
