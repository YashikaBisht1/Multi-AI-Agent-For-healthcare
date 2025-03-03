import streamlit as st
import base64
from wordcloud import WordCloud
from agents import AgentManager
from utils.logger import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    st.set_page_config(page_title="🚀 Multi-Agent AI System", layout="wide")

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1E88E5 0%, #1565C0 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(30,136,229,0.2);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        padding: 0.5rem 0;
        border-bottom: 2px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .validation-box {
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-header'> Multi-Agent AI System For Healthcare with Validation</div>",
                unsafe_allow_html=True)

    st.sidebar.title("Select Task")
    task = st.sidebar.selectbox("Choose a task:", [
        "🏥 Summarize Medical Text",
        "📄 Write and Refine Research Article",
        "🔒 Sanitize Medical Data (PHI)",
        "💬 AI Chatbot Assistant"
    ])

    agent_manager = AgentManager(max_retries=2, verbose=True)

    if task == "🏥 Summarize Medical Text":
        summarize_section(agent_manager)
    elif task == "📄 Write and Refine Research Article":
        write_and_refine_article_section(agent_manager)
    elif task == "🔒 Sanitize Medical Data (PHI)":
        sanitize_data_section(agent_manager)
    elif task == "💬 AI Chatbot Assistant":
        chatbot_section(agent_manager)


def summarize_section(agent_manager):
    st.markdown("<div class='sub-header'>🏥 Summarize Medical Text</div>", unsafe_allow_html=True)
    st.info("For accurate results and HIPAA compliance, PHI will always be removed before summarization.")
    text = st.text_area("📝 Enter medical text to summarize:", height=200)
    uploaded_file = st.file_uploader("📂 Upload a text file", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.getvalue().decode("utf-8")

    st.markdown("**Processing Order:**")
    st.markdown("1. **Sanitize Data (Remove PHI)**")
    st.markdown("2. **Summarize Medical Text**")
    st.markdown("3. **Validate Summary**")

    # Even if the user tries to choose a custom order, the system enforces the default pipeline.
    if st.button("✨ Process") and text:
        with st.spinner("🔄 Processing your input..."):
            try:
                # This method enforces: Sanitize → Summarize → Validate
                sanitized_text, summary, validation = agent_manager.process_summarization_pipeline(text)
                st.session_state["sanitized_text"] = sanitized_text
                st.session_state["summary"] = summary
                st.session_state["validation"] = validation

                st.markdown(f"<div class='result-box'><strong>✅ Sanitized Data:</strong><br>{sanitized_text}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='result-box'><strong>✅ Summary:</strong><br>{summary}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='validation-box'><strong>🔍 Validation:</strong><br>{validation}</div>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                logger.error(f"Summarization Pipeline Error: {e}")


def sanitize_data_section(agent_manager):
    st.markdown("<div class='sub-header'>🔒 Sanitize Medical Data (PHI)</div>", unsafe_allow_html=True)
    text = st.text_area("🔍 Paste the medical data to sanitize:", height=250)
    uploaded_file = st.file_uploader("📂 Upload a medical document", type=["txt", "csv"])
    if uploaded_file is not None:
        text = uploaded_file.getvalue().decode("utf-8")
    if st.button("🛡 Sanitize") and text:
        sanitize_agent = agent_manager.get_agent("sanitize_data")
        validator_agent = agent_manager.get_agent("sanitize_data_validator")
        with st.spinner("🔄 Removing PHI..."):
            sanitized_text = sanitize_agent.execute(text)
            st.session_state["sanitized_text"] = sanitized_text
            st.markdown(f"<div class='result-box'><strong>✅ Sanitized Data:</strong><br>{sanitized_text}</div>",
                        unsafe_allow_html=True)
        with st.spinner("🔍 Validating sanitization..."):
            validation = validator_agent.execute(original_data=text, sanitized_data=sanitized_text)
            st.session_state["sanitized_validation"] = validation
            st.markdown(f"<div class='validation-box'><strong>🧐 Validation Report:</strong><br>{validation}</div>",
                        unsafe_allow_html=True)
        download_results(st.session_state["sanitized_text"], st.session_state["sanitized_validation"],
                         "sanitized_data.txt")


def write_and_refine_article_section(agent_manager):
    st.markdown("<div class='sub-header'>📄 Write and Refine Research Article</div>", unsafe_allow_html=True)
    text = st.text_area("📝 Write or paste your research article:", height=300)
    uploaded_file = st.file_uploader("📂 Upload a document", type=["txt", "docx"])
    if uploaded_file is not None:
        text = uploaded_file.getvalue().decode("utf-8")
    if st.button("✍️ Write & Refine") and text:
        write_agent = agent_manager.get_agent("write_article")
        validator_agent = agent_manager.get_agent("write_article_validator")
        with st.spinner("🔄 Refining your article..."):
            try:
                refined_text = write_agent.execute(text)
                st.session_state["refined_text"] = refined_text
                st.markdown(f"<div class='result-box'><strong>✅ Refined Article:</strong><br>{refined_text}</div>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                return
        with st.spinner("🔍 Validating article..."):
            try:
                validation = validator_agent.execute(original_data=text, refined_data=refined_text)
                st.session_state["article_validation"] = validation
                st.markdown(f"<div class='validation-box'><strong>🧐 Validation Report:</strong><br>{validation}</div>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Validation Error: {e}")
                return
        if "refined_text" in st.session_state and "article_validation" in st.session_state:
            download_results(st.session_state["refined_text"], st.session_state["article_validation"],
                             "refined_article.txt")


def chatbot_section(agent_manager):
    st.markdown("<div class='sub-header'>💬 AI Chatbot Assistant</div>", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("💡 Ask me anything about medical research or AI:")
    if st.button("💬 Chat") and user_input:
        chatbot_agent = agent_manager.get_agent("chatbot")  # Ensure chatbot agent is implemented
        with st.spinner("🤖 Thinking..."):
            try:
                response = chatbot_agent.execute(user_input)
                st.session_state.chat_history.append(("🧑‍💻 You", user_input))
                st.session_state.chat_history.append(("🤖 AI", response))
            except Exception as e:
                st.error(f"⚠️ Chatbot Error: {e}")
    for role, message in st.session_state.chat_history:
        st.markdown(f"**{role}:** {message}")
    if st.button("🗑 Clear Chat History"):
        st.session_state.chat_history = []


def download_results(processed_text, validation_report, filename="results.txt"):
    """Generate a downloadable link for processed text and validation report."""
    combined_content = f"=== Processed Text ===\n{processed_text}\n\n=== Validation Report ===\n{validation_report}"
    b64 = base64.b64encode(combined_content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Click here to download results</a>'
    st.markdown(href, unsafe_allow_html=True)


def show_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


if __name__ == "__main__":
    main()
