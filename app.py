
import streamlit as st
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from agents import AgentManager
from utils.logger import logger
from dotenv import load_dotenv

import base64

# Load environment variables
load_dotenv()

from streamlit_lottie import st_lottie
import json

def main():
    st.set_page_config(page_title="Healthcare AI Hub", layout="wide")
    if st.session_state.view == "home":
        show_home()
    else:
        show_agent_view(st.session_state.view)

# Track current view
if "view" not in st.session_state:
    st.session_state.view = "home"

# Navigation functions
def go_home():
    st.session_state.view = "home"

def go_to_agent(agent_name):
    st.session_state.view = agent_name

 # Card styling
def render_card(title, description, button_label, key, agent_name):
    st.markdown(f"""
    <style>
    .card-{key} {{
        background: linear-gradient(135deg, #1e1e1e, #111);
        padding: 25px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 4px 30px rgba(0,0,0,0.5);
        text-align: center;
        transition: transform 0.2s ease;
        cursor: pointer;
        margin-bottom: 20px;
    }}
    .card-{key}:hover {{
        transform: scale(1.02);
        box-shadow: 0 6px 40px rgba(0, 255, 150, 0.4);
    }}
    </style>
    <div class='card-{key}'>
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

    # Button inside container
    st.button(button_label, key=key, on_click=lambda: go_to_agent(agent_name))

# Home page with 4 cards
def show_home():
    st.title("👨‍⚕️ Multi-Agent AI Healthcare System")

    col1, col2 = st.columns(2)
    with col1:
        render_card("📄 Summarizer", "Summarizes long medical articles.", "Launch Summarizer", "summarizer_btn",
                        "summarizer")
    with col2:
        render_card("📝 Article Refiner", "Refines and enhances content.", "Launch Refiner", "refiner_btn",
                        "refiner")

    col3, col4 = st.columns(2)
    with col3:
        render_card("🔒 Data Sanitizer", "Cleans and anonymizes data.", "Launch Sanitizer", "sanitizer_btn",
                        "sanitizer")
    with col4:
        render_card("🤖 Medical Chatbot", "Chat with an AI medical assistant.", "Launch Chatbot", "chatbot_btn",
                        "chatbot")


agent_manager = AgentManager(max_retries=2, verbose=True)


# Your agent implementations here
def summarizer():
    st.header("📝 Summarizer Agent")
    summarize_section(agent_manager)
    st.button("🔙 Back to Home", on_click=go_home)


def refiner():
    st.header("🔧 Refiner Agent")
    write_and_refine_article_section(agent_manager)
    st.button("🔙 Back to Home", on_click=go_home)


def sanitizer():
    st.header("🔒 Sanitizer Agent")
    sanitize_data_section(agent_manager)
    st.button("🔙 Back to Home", on_click=go_home)


def chatbot():
    st.header("💬 Chatbot Agent")
    chatbot_section(agent_manager)
    st.button("🔙 Back to Home", on_click=go_home)

    # Agent views
def show_agent_view(agent):
    if agent == "summarizer":
        summarizer()
    elif agent == "refiner":
        refiner()
    elif agent == "sanitizer":
        sanitizer()
    elif agent == "chatbot":
        chatbot()


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
                show_wordcloud(refined_text)  # Show word cloud for refined article
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                logger.error(f"WriteArticleAgent Error: {e}")
                return

        with st.spinner("🔍 Validating article..."):
            try:
                validation = validator_agent.execute(original_data=text, refined_data=refined_text)
                st.session_state["article_validation"] = validation
                st.markdown(f"<div class='validation-box'><strong>🧐 Validation Report:</strong><br>{validation}</div>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Validation Error: {e}")
                logger.error(f"WriteArticleValidatorAgent Error: {e}")
                return

        if "refined_text" in st.session_state and "article_validation" in st.session_state:
            download_results(st.session_state["refined_text"], st.session_state["article_validation"],
                             "refined_article.txt")


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
            try:
                sanitized_text = sanitize_agent.execute(text)
                st.session_state["sanitized_text"] = sanitized_text
                st.markdown(f"<div class='result-box'><strong>✅ Sanitized Data:</strong><br>{sanitized_text}</div>",
                            unsafe_allow_html=True)
                show_wordcloud(sanitized_text)  # Generate a word cloud for sanitized data
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                logger.error(f"SanitizeDataAgent Error: {e}")
                return

        with st.spinner("🔍 Validating sanitization..."):
            try:
                validation = validator_agent.execute(original_data=text, sanitized_data=sanitized_text)
                st.session_state["sanitized_validation"] = validation
                st.markdown(f"<div class='validation-box'><strong>🧐 Validation Report:</strong><br>{validation}</div>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Validation Error: {e}")
                logger.error(f"SanitizeDataValidatorAgent Error: {e}")
                return

        if "sanitized_text" in st.session_state and "sanitized_validation" in st.session_state:
            download_results(st.session_state["sanitized_text"], st.session_state["sanitized_validation"],
                             "sanitized_data.txt")
def chatbot_section(agent_manager):
    st.markdown("<div class='sub-header'>💬 AI Chatbot Assistant</div>", unsafe_allow_html=True)

    def load_lottie_file(filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    lottie_chatbot = load_lottie_file("animations/chatbot.json")

    # Create two columns: left for animation, right for content
    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie(lottie_chatbot, height=400, width=400, key="chatbot")

    with col2:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Toggle model selection
        use_biogpt = st.radio("🤖 Choose AI Model:", ["BioGPT", "LLaMA/Ollama"], horizontal=True) == "BioGPT"

        user_input = st.text_input("💡 Ask me anything about medical research or AI:")

        if st.button("💬 Chat") and user_input:
            chatbot_agent = agent_manager.get_agent("chatbot", use_biogpt=use_biogpt)

            with st.spinner("🤖 Thinking..."):
                try:
                    response = chatbot_agent.execute(user_input)
                    st.session_state.chat_history.append(("🧑‍💻 You", user_input))
                    st.session_state.chat_history.append(("🤖 AI", response))
                except Exception as e:
                    st.error(f"⚠️ Chatbot Error: {e}")
                    logger.error(f"ChatbotAgent Error: {e}")

        for role, message in st.session_state.chat_history:
            st.markdown(f"**{role}:** {message}")

        if st.button("🗑 Clear Chat History"):
            st.session_state.chat_history = []


def summarize_section(agent_manager):
    st.markdown("<div class='sub-header'>🏥 Summarize Medical Text</div>", unsafe_allow_html=True)
    text = st.text_area("📝 Enter medical text to summarize:", height=200)
    uploaded_file = st.file_uploader("📂 Upload a text file", type=["txt"])

    if uploaded_file is not None:
        text = uploaded_file.getvalue().decode("utf-8")

    if st.button("✨ Summarize") and text:
        main_agent = agent_manager.get_agent("summarize")
        validator_agent = agent_manager.get_agent("summarize_validator")

        with st.spinner("🔄 Summarizing..."):
            try:
                summary = main_agent.execute(text)
                st.session_state["summary"] = summary
                st.markdown(f"<div class='result-box'><strong>✅ Summary:</strong><br>{summary}</div>", unsafe_allow_html=True)
                show_wordcloud(text)
            except Exception as e:
                st.error(f"⚠️ Error during summarization: {e}")
                logger.error(f"SummarizeAgent Error: {e}")
                return

        with st.spinner("🔄 Validating summary..."):
            try:
                validation, ai_score = validator_agent.execute(original_text=text, summary=summary)
                st.session_state["validation"] = validation
                st.markdown(f"<div class='validation-box'><strong>🔍 Validation Report:</strong><br>{validation}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rating-box'><strong>🤖 AI Rating:</strong> {ai_score:.1f} / 5</div>", unsafe_allow_html=True)

                # Human input
                human_score = st.number_input("🧠 Your Rating (1.0 to 5.0):", min_value=1.0, max_value=5.0, step=0.1, key="rating_input")
                final_summary = summary

                if human_score:
                    avg_score = round((ai_score + human_score) / 2, 1)
                    st.session_state["validation_rating"] = human_score
                    st.markdown(f"<div class='rating-box'><strong>📊 Average Rating:</strong> {avg_score} / 5</div>", unsafe_allow_html=True)

                    # Improve if needed
                    if avg_score < 3.5:
                        with st.spinner("🔁 Improving summary..."):
                            try:
                                improved_prompt = (
                                    f"Improve the following summary based on the original text. "
                                    f"Ensure it's more concise, accurate, and medically appropriate.\n\n"
                                    f"Original Text:\n{text}\n\nSummary:\n{summary}"
                                )
                                improved_summary = validator_agent.call_llama([
                                    {"role": "system", "content": "You are a medical summarization improver."},
                                    {"role": "user", "content": improved_prompt}
                                ])
                                final_summary = improved_summary
                                st.markdown(f"<div class='result-box'><strong>🔁 Improved Summary:</strong><br>{improved_summary}</div>", unsafe_allow_html=True)
                            except Exception as e:
                                st.warning(f"⚠️ Couldn't improve summary: {e}")

                # Download report
                download_summary_report(
                    original_text=text,
                    summary=final_summary,
                    validation_report=validation,
                    ai_rating=ai_score,
                    human_rating=human_score
                )


            except Exception as e:
                st.error(f"⚠️ Validation Error: {e}")
                logger.error(f"SummarizeValidatorAgent Error: {e}")

from io import BytesIO
from datetime import datetime

def download_summary_report(original_text, summary, validation_report, ai_rating, human_rating):
    """
    Creates and enables downloading of a summary validation report.
    Includes original text, final summary, validation notes, and ratings.
    """
    avg_rating = round((ai_rating + human_rating) / 2, 1)

    report = f"""🧾 MEDICAL SUMMARY REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*60}

📄 ORIGINAL TEXT:
{original_text.strip()}

{"="*60}
📝 FINAL SUMMARY:
{summary.strip()}

{"="*60}
🔍 VALIDATION REPORT:
{validation_report.strip()}

{"="*60}
📊 RATINGS:
🤖 AI Rating     : {ai_rating} / 5
🧠 Human Rating  : {human_rating} / 5
📈 Average Rating: {avg_rating} / 5
"""

    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)

    filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button(
        label="⬇️ Download Summary Report",
        data=buffer,
        file_name=filename,
        mime="text/plain"
    )

def download_results(processed_text, validation_report, filename="results.txt"):
    """Generate a downloadable link for processed text and validation report."""
    combined_content = f"=== Processed Text ===\n{processed_text}\n\n=== Validation Report ===\n{validation_report}"

    b64 = base64.b64encode(combined_content.encode()).decode()  # Encode the text file
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Click here to download results</a>'

    st.markdown(href, unsafe_allow_html=True)


def show_wordcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=400, max_words=25, background_color='white', colormap='Set2',
                      collocations=False, stopwords=STOPWORDS).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


if __name__ == "__main__":
    main()

