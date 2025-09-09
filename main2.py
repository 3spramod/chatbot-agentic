import streamlit as st
import requests
import base64

# ==================== CONFIG ====================
BACKEND_BASE_URL = "http://127.0.0.1:5000"  # Flask backend
FAVICON_PATH = "favicon.png"

# Function to convert image to Base64
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

favicon_base64 = image_to_base64(FAVICON_PATH)

st.set_page_config(page_title="Prism Multi-Workspace", page_icon=FAVICON_PATH, layout="centered")

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "workspace" not in st.session_state:
    st.session_state.workspace = "file1"
if st.session_state.workspace not in st.session_state.messages:
    st.session_state.messages[st.session_state.workspace] = []

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Workspace Selector")
workspace = st.sidebar.radio("Choose a file:", ["file1", "file2", "file3"])

# Reset messages if workspace changed
if workspace != st.session_state.workspace:
    st.session_state.workspace = workspace
    if workspace not in st.session_state.messages:
        st.session_state.messages[workspace] = []

    # Call ingest for new workspace
    with st.spinner(f"🔄 Loading {workspace}..."):
        resp = requests.post(f"{BACKEND_BASE_URL}/ingest/{workspace}")
        if resp.status_code == 200:
            st.sidebar.success(f"✅ {workspace} ready!")
        else:
            st.sidebar.error(f"⚠️ Ingestion failed: {resp.text}")

# ==================== HEADER ====================
st.markdown(
    f"""
     <h1 style='text-align: center;'>
        <img src='data:image/png;base64,{favicon_base64}' width='28' style='vertical-align:middle; margin-right: 1px; height: 52px; width: 40px;'/>
        Prism Multi-Workspace
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="text-align:center; color:gray; font-size:30px; margin-bottom:20px;">
        You are working on: <b>{workspace}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== CHAT UI ====================
for msg in st.session_state.messages[workspace]:
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
    st.session_state.messages[workspace].append({"role": "user", "content": user_input})

    try:
        response = requests.post(f"{BACKEND_BASE_URL}/query/{workspace}", json={"query": user_input})
        if response.status_code == 200:
            data = response.json()
            bot_reply = data.get("response", "⚠️ No answer received.")
        else:
            bot_reply = f"⚠️ Backend error {response.status_code}: {response.text}"
    except Exception as e:
        bot_reply = f"⚠️ Connection error: {str(e)}"

    st.session_state.messages[workspace].append({"role": "bot", "content": bot_reply})
    st.rerun()
