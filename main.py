import streamlit as st
import requests
import base64

# ==================== CONFIG ====================
BACKEND_URL = "http://127.0.0.1:5000/query"  # Adjust if running elsewhere
BACKEND_INGEST_URL = "http://127.0.0.1:5000/ingest"  # 👈 Add ingest API
FAVICON_PATH = "favicon.png"  # Your local favicon

# Function to convert image file to Base64
def image_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

favicon_base64 = image_to_base64(FAVICON_PATH)

st.set_page_config(page_title="Prism1", page_icon=FAVICON_PATH, layout="centered")

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []



# --- Run ingestion only once when app loads ---
if "ingested" not in st.session_state:
    with st.spinner("🔄 Loading knowledge base..."):
        try:
            resp = requests.post(BACKEND_INGEST_URL)  # 👈 Call ingest
            if resp.status_code == 200:
                st.session_state.ingested = True
                st.success("✅ Knowledge base ready!")
            else:
                st.error(f"⚠️ Ingestion failed: {resp.status_code}")
                st.session_state.ingested = False
        except Exception as e:
            st.error(f"⚠️ Could not connect to ingestion API: {e}")
            st.session_state.ingested = False

# ==================== HEADER ====================
st.markdown(
    f"""
     <h1 style='text-align: center;'>
        <img src='data:image/png;base64,{favicon_base64}' width='28' style='vertical-align:middle; margin-right:8px;'/>
        Prism1
    </h1>
    """,
    unsafe_allow_html=True,
)

# --- Chatbot Description ---
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:30px; margin-bottom:20px;">
        Explore data with ease with your AI assistant : Prism
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== CHAT UI ====================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin:8px 0;">
                <div style="background-color:#DCF8C6; padding:10px 15px; border-radius:12px; max-width:70%; text-align:right; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                    <b>🧑 You</b><br>{msg["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-start; margin:8px 0;">
                <div style="background-color:#F1F0F0; padding:10px 15px; border-radius:12px; max-width:70%; text-align:left; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                    <b><img src='data:image/png;base64,{favicon_base64}' width='16' style='vertical-align:middle; margin-right:4px;'/> Prism</b><br>{msg["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==================== INPUT BOX ====================
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        # Send query to backend
        response = requests.post(BACKEND_URL, json={"query": user_input})
        if response.status_code == 200:
            data = response.json()
            bot_reply = data.get("response", "⚠️ No answer received.")
        else:
            bot_reply = f"⚠️ Backend error {response.status_code}: {response.text}"

    except Exception as e:
        bot_reply = f"⚠️ Connection error: {str(e)}"

    # Append bot reply
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

    # Rerun to refresh UI
    st.rerun()
