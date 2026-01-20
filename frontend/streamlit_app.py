import streamlit as st
import requests
import uuid

# --- Configuration & UI Setup ---
st.set_page_config(page_title="Civic Copilot", page_icon="🏛️", layout="wide")

API_URL = "http://localhost:8000/api/v1/chat"

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🏛️ Civic Copilot")
st.markdown("Your agentic guide for navigating government processes and civic inquiries.")
st.divider()

# --- Sidebar for Metadata & Settings ---
with st.sidebar:
    st.header("Session Info")
    st.caption(f"ID: {st.session_state.session_id}")
    if st.button("Reset Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.info("This system uses verified official policies to guide your civic actions.")

# --- Chat Display ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display extra structured data if it's an assistant response
        if message["role"] == "assistant" and "data" in message:
            data = message["data"]
            
            # 1. Action Steps Card
            if data.get("action_data") and data["action_data"].get("immediate_steps"):
                with st.expander("✅ Recommended Action Steps", expanded=True):
                    for step in data["action_data"]["immediate_steps"]:
                        st.write(f"**{step['order']}. {step['action']}**")
                        st.write(f"{step['details']}")
                        if step.get("link"):
                            st.link_button("Go to Portal", step["link"])

            # 2. Source Citations
            if data.get("policy_data") and data["policy_data"].get("sources"):
                with st.expander("📚 Official Sources & Citations"):
                    for src in data["policy_data"]["sources"]:
                        st.markdown(f"- [{src['title']}]({src['url']})")

# --- Chat Input & Logic ---
if prompt := st.chat_input("How can I help you today? (e.g., Passport delays, Voter registration)"):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Backend API
    with st.chat_message("assistant"):
        with st.spinner("Orchestrating agents..."):
            try:
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                
                result = response.json()
                answer = result["answer_text"]
                
                # Display Answer
                st.markdown(answer)
                
                # Show Verification Badge
                if result.get("is_verified"):
                    st.caption("🛡️ Verified against official documents.")
                else:
                    st.warning(f"⚠️ {result.get('risk_disclaimer', 'Guidance not fully verified.')}")

                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "data": result
                })
                
                # Trigger a rerun to show the newly added expanders properly
                st.rerun()

            except Exception as e:
                st.error(f"Failed to connect to Civic Copilot backend. Error: {e}")