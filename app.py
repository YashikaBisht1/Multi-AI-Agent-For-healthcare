import streamlit as st

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from agents import AgentManager
from utils.logger import logger
from dotenv import load_dotenv
from agents.agent_base import AgentBase
from io import BytesIO
from datetime import datetime
import numpy as np
import json
import os


# Load environment variables
load_dotenv()

# Cache the wordcloud generation
@st.cache_data
def generate_wordcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=400, max_words=25, 
                         background_color='white', colormap='Set2',
                         collocations=False, stopwords=STOPWORDS).generate(text)
    return wordcloud

# Cache the agent manager initialization
@st.cache_resource
def get_agent_manager():
    return AgentManager(max_retries=2, verbose=True)

def show_wordcloud(text):
    wordcloud = generate_wordcloud(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def main():
    st.set_page_config(page_title="Multi-Agent AI System", layout="wide")

    st.markdown("""
    <style>
    html, body, [class^="css"]  {
        background-color: #f5f8ff;
    }
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
    .task-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .result-box, .validation-box, .rating-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .result-box {
        border-left: 4px solid #43a047;
    }
    .validation-box {
        border-left: 4px solid #fb8c00;
    }
    .rating-box {
        border-left: 4px solid #3949ab;
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
    .stTextArea>div>div {
        border-radius: 10px;
        border: 2px solid #bbdefb;
    }
    .sidebar-content {
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-header'> Multi-Agent AI System For Healthcare with Validation</div>", unsafe_allow_html=True)

    st.sidebar.title("Select Task")
    task = st.sidebar.selectbox("Choose a task:", [
        "🏥 Summarize Medical Text",
        "📄 Write and Refine Research Article",
        "🔒 Sanitize Medical Data (PHI)",
        "💬 AI Chatbot Assistant"
    ])

    agent_manager = get_agent_manager()

    if task == "🏥 Summarize Medical Text":
        summarize_section(agent_manager)
    elif task == "📄 Write and Refine Research Article":
        write_and_refine_article_section(agent_manager)
    elif task == "🔒 Sanitize Medical Data (PHI)":
        sanitize_data_section(agent_manager)
    elif task == "💬 AI Chatbot Assistant":
        chatbot_section(agent_manager)


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
                validation_response, ai_rating, _ = validator_agent.execute(topic=text, article=refined_text)
                st.session_state["article_validation"] = validation_response
                st.session_state["article_ai_score"] = ai_rating
                st.markdown(f"<div class='validation-box'><strong>🧐 Validation Report:</strong><br>{validation_response}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rating-box'><strong>🤖 AI Rating:</strong> {ai_rating:.1f} / 5</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Validation Error: {e}")
                logger.error(f"WriteArticleValidatorAgent Error: {e}")
                return

    if "article_validation" in st.session_state and "refined_text" in st.session_state and "article_ai_score" in st.session_state:
        human_score = st.number_input("🧠 Your Rating (1.0 to 5.0):", min_value=1.0, max_value=5.0, step=0.1, key="article_rating_input")
        if st.button("Submit Article Rating"):
            ai_score = st.session_state["article_ai_score"]
            refined_text = st.session_state["refined_text"]
            validation_response = st.session_state["article_validation"]
            avg_score = round((ai_score + human_score) / 2, 1)
            st.session_state["article_validation_rating"] = human_score
            st.markdown(f"<div class='rating-box'><strong>📊 Average Rating:</strong> {avg_score} / 5</div>", unsafe_allow_html=True)
            # Store feedback with human rating
            validator_agent = agent_manager.get_agent("write_article_validator")
            validator_agent.store_feedback(text, refined_text, ai_score, human_score)
            store_feedback_json("write_article", {
                "original": text,
                "refined": refined_text,
                "ai_rating": ai_score,
                "human_rating": human_score,
                "validation": validation_response
            })
            improved_article = None
            if avg_score < 3.5:
                with st.spinner("🔁 Improving article..."):
                    try:
                        improved_prompt = (
                            f"Improve the following research article based on the original. "
                            f"Ensure it's more concise, accurate, and medically appropriate.\n\n"
                            f"Original Article:\n{text}\n\nRefined Article:\n{refined_text}"
                        )
                        improved_article = validator_agent.call_llama([
                            {"role": "system", "content": "You are a research article improver."},
                            {"role": "user", "content": improved_prompt}
                        ])
                        st.markdown(f"<div class='result-box'><strong>🔁 Improved Article:</strong><br>{improved_article}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"⚠️ Couldn't improve article: {e}")
            download_article_report(text, refined_text, validation_response, ai_score, human_score, improved_article)


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
                validation_response, ai_score, _, _ = validator_agent.execute(original_data=text, sanitized_data=sanitized_text)
                st.session_state["sanitized_validation"] = validation_response
                st.session_state["sanitize_ai_score"] = ai_score
                st.markdown(f"<div class='validation-box'><strong>🧐 Validation Report:</strong><br>{validation_response}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rating-box'><strong>🤖 AI Rating:</strong> {ai_score:.1f} / 5</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Validation Error: {e}")
                logger.error(f"SanitizeDataValidatorAgent Error: {e}")
                return

    if "sanitized_validation" in st.session_state and "sanitized_text" in st.session_state and "sanitize_ai_score" in st.session_state:
        human_score = st.number_input("🧠 Your Rating (1.0 to 5.0):", min_value=1.0, max_value=5.0, step=0.1, key="sanitize_rating_input")
        if st.button("Submit Sanitize Rating"):
            ai_score = st.session_state["sanitize_ai_score"]
            sanitized_text = st.session_state["sanitized_text"]
            validation_response = st.session_state["sanitized_validation"]
            avg_score = round((ai_score + human_score) / 2, 1)
            st.session_state["sanitized_validation_rating"] = human_score
            st.markdown(f"<div class='rating-box'><strong>📊 Average Rating:</strong> {avg_score} / 5</div>", unsafe_allow_html=True)
            # Store feedback with human rating
            validator_agent = agent_manager.get_agent("sanitize_data_validator")
            validator_agent.store_feedback(text, sanitized_text, ai_score, human_score)
            store_feedback_json("sanitize", {
                "original": text,
                "sanitized": sanitized_text,
                "ai_rating": ai_score,
                "human_rating": human_score,
                "validation": validation_response
            })
            improved_sanitized = None
            if avg_score < 3.5:
                with st.spinner("🔁 Improving sanitized data..."):
                    try:
                        improved_prompt = (
                            f"Improve the following sanitized medical data based on the original. "
                            f"Ensure all PHI is masked and the data is more accurate.\n\n"
                            f"Original Data:\n{text}\n\nSanitized Data:\n{sanitized_text}"
                        )
                        improved_sanitized = validator_agent.call_llama([
                            {"role": "system", "content": "You are a medical data sanitizer improver."},
                            {"role": "user", "content": improved_prompt}
                        ])
                        st.markdown(f"<div class='result-box'><strong>🔁 Improved Sanitized Data:</strong><br>{improved_sanitized}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"⚠️ Couldn't improve sanitized data: {e}")
            download_sanitize_report(text, sanitized_text, validation_response, ai_score, human_score, improved_sanitized)

from PIL import Image
import base64


def chatbot_section(agent_manager):
    st.markdown("""
        <style>
        .chat-bg {
            background-image: url('https://images.unsplash.com/photo-1607746882042-944635dfe10e');
            background-size: cover;
            background-position: center;
            padding: 0;
            margin: 0;
        }

        .chat-overlay {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 12px;
        }

        .sub-header {
            font-size: 32px;
            color: white;
            text-align: center;
            font-weight: bold;
            margin-bottom: 2rem;
        }

        .chat-container {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
        }

        .chat-left {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .chat-right {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
        }

        .message-block {
            margin-bottom: 1rem;
            padding: 0.75rem;
            border-radius: 6px;
        }

        .user-msg {
            background-color: #dbeafe;
        }

        .ai-msg {
            background-color: #ede9fe;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='chat-bg'><div class='chat-overlay'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>💬 AI Chatbot Assistant</div>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("💡 Ask me anything about medical research or AI:")

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Left: Chatbot Image
    with st.container():
        st.markdown("<div class='chat-left'>", unsafe_allow_html=True)
        chatbot_image = Image.open("chatbot_image.png")
        st.image(chatbot_image, width=300)
        st.markdown("</div>", unsafe_allow_html=True)

    # Right: Chat Interface
    with st.container():
        st.markdown("<div class='chat-right'>", unsafe_allow_html=True)

        if st.button("💬 Chat") and user_input:
            chatbot_agent = agent_manager.get_agent("chatbot")

            with st.spinner("🤖 Thinking..."):
                try:
                    response = chatbot_agent.execute(user_input)
                    st.session_state.chat_history.append(("🧑‍💻 You", user_input))
                    st.session_state.chat_history.append(("🤖 AI", response))
                except Exception as e:
                    st.error(f"⚠️ Chatbot Error: {e}")

        for role, message in st.session_state.chat_history:
            css_class = "user-msg" if "You" in role else "ai-msg"
            st.markdown(f"<div class='message-block {css_class}'><strong>{role}:</strong> {message}</div>", unsafe_allow_html=True)

        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []

        st.markdown("</div>", unsafe_allow_html=True)  # Close .chat-right

    st.markdown("</div>", unsafe_allow_html=True)  # Close .chat-container
    st.markdown("</div></div>", unsafe_allow_html=True)  # Close overlays





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
                validation_response, ai_score, _, _ = validator_agent.execute(original_text=text, summary=summary)
                st.session_state["validation"] = validation_response
                st.session_state["ai_score"] = ai_score
                st.markdown(f"<div class='validation-box'><strong>🔍 Validation Report:</strong><br>{validation_response}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rating-box'><strong>🤖 AI Rating:</strong> {ai_score:.1f} / 5</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Validation Error: {e}")
                logger.error(f"SummarizeValidatorAgent Error: {e}")
                return

    if "validation" in st.session_state and "summary" in st.session_state and "ai_score" in st.session_state:
        human_score = st.number_input("🧠 Your Rating (1.0 to 5.0):", min_value=1.0, max_value=5.0, step=0.1, key="rating_input")
        if st.button("Submit Rating"):
            ai_score = st.session_state["ai_score"]
            summary = st.session_state["summary"]
            validation_response = st.session_state["validation"]
            avg_score = round((ai_score + human_score) / 2, 1)
            st.session_state["validation_rating"] = human_score
            st.markdown(f"<div class='rating-box'><strong>📊 Average Rating:</strong> {avg_score} / 5</div>", unsafe_allow_html=True)
            # Store feedback with human rating
            validator_agent = agent_manager.get_agent("summarize_validator")
            validator_agent.store_feedback(text, summary, ai_score, human_score)
            store_feedback_json("summarize", {
                "original": text,
                "summary": summary,
                "ai_rating": ai_score,
                "human_rating": human_score,
                "validation": validation_response
            })
            # Optionally allow improvement and download as before...
            # Download report
            download_summary_report(
                original_text=text,
                summary=summary,
                validation_report=validation_response,
                ai_rating=ai_score,
                human_rating=human_score
            )


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

def download_sanitize_report(original_data, sanitized_data, validation_report, ai_rating, human_rating, improved_sanitized=None):
    """
    Creates and enables downloading of a sanitization report.
    Includes original data, sanitized output, validation notes, ratings, and improved sanitized content if available.
    """
    avg_rating = round((ai_rating + human_rating) / 2, 1)
    report = f"""🛡 SANITIZED DATA REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*60}

📄 ORIGINAL DATA:
{original_data.strip()}

{"="*60}
🔒 SANITIZED OUTPUT:
{sanitized_data.strip()}

{"="*60}
🔍 VALIDATION REPORT:
{validation_report.strip()}

{"="*60}
📊 RATINGS:
🤖 AI Rating     : {ai_rating} / 5
🧠 Human Rating  : {human_rating} / 5
📈 Average Rating: {avg_rating} / 5
"""
    if improved_sanitized:
        report += f"\n{'='*60}\n✨ IMPROVED SANITIZED OUTPUT:\n{improved_sanitized.strip()}\n"

    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)

    filename = f"sanitized_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button(
        label="⬇️ Download Sanitization Report",
        data=buffer,
        file_name=filename,
        mime="text/plain"
    )

def download_article_report(original_article, refined_article, validation_report, ai_rating, human_rating, improved_article=None):
    """
    Creates and enables downloading of an article writing/refinement report.
    Includes original article, refined version, validation, ratings, and improved article if available.
    """
    avg_rating = round((ai_rating + human_rating) / 2, 1)
    report = f"""📝 RESEARCH ARTICLE REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*60}

🧾 ORIGINAL ARTICLE:
{original_article.strip()}

{"="*60}
✍️ REFINED ARTICLE:
{refined_article.strip()}

{"="*60}
🔍 VALIDATION REPORT:
{validation_report.strip()}

{"="*60}
📊 RATINGS:
🤖 AI Rating     : {ai_rating} / 5
🧠 Human Rating  : {human_rating} / 5
📈 Average Rating: {avg_rating} / 5
"""
    if improved_article:
        report += f"\n{'='*60}\n✨ IMPROVED ARTICLE:\n{improved_article.strip()}\n"

    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)

    filename = f"article_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button(
        label="⬇️ Download Article Report",
        data=buffer,
        file_name=filename,
        mime="text/plain"
    )

FEEDBACK_FILE = "feedback_store.json"

def store_feedback_json(section, feedback_entry):
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}
    if section not in data:
        data[section] = []
    data[section].append(feedback_entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
