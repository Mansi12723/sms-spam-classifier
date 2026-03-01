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
    layout="wide"
)

# ---------------- GMAIL STYLE CSS ----------------
st.markdown("""
<style>

/* Hide Streamlit default header/footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Page Background */
.stApp {
    background-color: #f6f8fc;
}

/* Top Header */
.gmail-header {
    background-color: white;
    padding: 15px 25px;
    font-size: 22px;
    font-weight: bold;
    border-bottom: 1px solid #ddd;
}

/* Sidebar */
.sidebar-box {
    background-color: white;
    padding: 15px;
    border-right: 1px solid #ddd;
    height: 100vh;
}

/* Sidebar item */
.menu-item {
    padding: 10px;
    font-size: 16px;
    cursor: pointer;
}

.menu-item:hover {
    background-color: #e8f0fe;
    border-radius: 6px;
}

/* Email card */
.email-card {
    background-color: white;
    padding: 12px;
    margin-bottom: 8px;
    border-bottom: 1px solid #eee;
}

/* Button */
.stButton>button {
    background-color: #1a73e8;
    color: white;
    border-radius: 4px;
    border: none;
    padding: 8px 16px;
}

/* Text area */
.stTextArea textarea {
    background-color: white;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="gmail-header">Email Spam Detection</div>', unsafe_allow_html=True)

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

# ---------------- SESSION STATE ----------------
if "inbox" not in st.session_state:
    st.session_state.inbox = []

if "spam" not in st.session_state:
    st.session_state.spam = []

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 4])

# -------- LEFT SIDEBAR --------
with col1:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="menu-item"><b>📥 Inbox ({len(st.session_state.inbox)})</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="menu-item"><b>🚫 Spam ({len(st.session_state.spam)})</b></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------- MAIN CONTENT --------
with col2:

    st.subheader("Compose / Scan Email")
    input_text = st.text_area("Enter email content", height=120)

    if st.button("Analyze"):
        if input_text.strip() == "":
            st.warning("Please enter email content.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(1)

            transformed = transform_text(input_text)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.session_state.spam.append(input_text)
            else:
                st.session_state.inbox.append(input_text)

    # -------- DISPLAY INBOX --------
    st.markdown("### Inbox")
    for mail in st.session_state.inbox:
        st.markdown(f'<div class="email-card">{mail}</div>', unsafe_allow_html=True)

    # -------- DISPLAY SPAM --------
    st.markdown("### Spam")
    for mail in st.session_state.spam:
        st.markdown(f'<div class="email-card">{mail}</div>', unsafe_allow_html=True)
