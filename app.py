import streamlit as st
import pickle
import nltk
import string
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Gmail Clone - Spam Detection", layout="wide")

# ---------------- GMAIL CSS ----------------
st.markdown("""
<style>

/* Hide default */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Whole background */
.stApp {
    background-color: #f1f3f4;
}

/* Top Navbar */
.topbar {
    background: #ffffff;
    padding: 12px 20px;
    border-bottom: 1px solid #ddd;
    font-size: 20px;
    font-weight: 500;
}

/* Sidebar */
.sidebar {
    background: #ffffff;
    height: 100vh;
    padding: 15px;
    border-right: 1px solid #ddd;
}

/* Compose Button */
.compose-btn {
    background: #c2e7ff;
    padding: 10px;
    border-radius: 20px;
    text-align: center;
    font-weight: 500;
    margin-bottom: 20px;
}

/* Menu item */
.menu-item {
    padding: 10px;
    font-size: 15px;
    cursor: pointer;
    border-radius: 20px;
}

.menu-item:hover {
    background-color: #e8f0fe;
}

/* Email List */
.email-row {
    background: white;
    padding: 12px;
    border-bottom: 1px solid #eee;
}

/* Button */
.stButton>button {
    background-color: #1a73e8;
    color: white;
    border-radius: 4px;
    border: none;
}

/* Text Area */
.stTextArea textarea {
    background-color: white;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TOP BAR ----------------
st.markdown('<div class="topbar">📧 Gmail - Spam Detection System</div>', unsafe_allow_html=True)

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

# ---------------- SESSION ----------------
if "inbox" not in st.session_state:
    st.session_state.inbox = []

if "spam" not in st.session_state:
    st.session_state.spam = []

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Inbox"

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 4])

# -------- LEFT SIDEBAR --------
with col1:
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    st.markdown('<div class="compose-btn">✏ Compose</div>', unsafe_allow_html=True)

    if st.button("📥 Inbox"):
        st.session_state.active_tab = "Inbox"

    if st.button("🚫 Spam"):
        st.session_state.active_tab = "Spam"

    st.markdown('</div>', unsafe_allow_html=True)

# -------- MAIN CONTENT --------
with col2:

    # Compose / Scan section
    st.subheader("Compose Email")
    input_text = st.text_area("Write your email", height=120)

    if st.button("Send / Analyze"):
        if input_text.strip() != "":
            with st.spinner("Checking for spam..."):
                time.sleep(1)

            transformed = transform_text(input_text)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.session_state.spam.append(input_text)
            else:
                st.session_state.inbox.append(input_text)

    st.markdown("---")

    # Display selected tab
    st.subheader(st.session_state.active_tab)

    emails_to_show = (
        st.session_state.inbox
        if st.session_state.active_tab == "Inbox"
        else st.session_state.spam
    )

    for mail in emails_to_show:
        st.markdown(f'<div class="email-row">{mail}</div>', unsafe_allow_html=True)
