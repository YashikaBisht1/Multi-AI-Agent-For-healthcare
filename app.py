import streamlit as st

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from agents import AgentManager
from utils.logger import logger
from dotenv import load_dotenv

import base64

# Load environment variables
load_dotenv()


def main():
    st.set_page_config(page_title="Multi-Agent AI System", layout="wide")

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
    .task-container {
        background-color: #F8F9FA;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    .rating-box {
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        background-color: #E3F2FD;
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
        border-radius: 8px;
        border: 2px solid #E3F2FD;
    }
    .sidebar-content {
        padding: 1rem;
        background-color: #F8F9FA;
        border-radius: 8px;
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

class WriteArticleValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="WriteArticleValidatorAgent", max_retries=max_retries, verbose=verbose)
        self.validation_history = []  # Store validation feedback
        self.temperature = 0.7
        self.max_tokens = 512

    def execute(self, topic, article):
        """
        Validates the quality and completeness of a research article.
        """
        system_message = "You are an AI assistant that validates research articles."
        user_content = (
            "Given the topic and the article, assess whether the article comprehensively covers the topic, "
            "follows a logical structure, and maintains academic standards.\n"
            "Provide a brief analysis and rate the article on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
            f"Topic:\n{topic}\n\n"
            f"Article:\n{article}\n\n"
            "Validation Report:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        validation_response = self.call_llama(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        ai_rating = self.extract_validation_score(validation_response)
        human_rating = self.get_human_feedback(validation_response)

        average_score = (ai_rating + human_rating) / 2

        self.store_feedback(topic, article, ai_rating, human_rating)
        self.optimize_with_rl()

        return validation_response, average_score

    def extract_validation_score(self, response):
        """
        Extracts the AI-generated rating from the response (1-5 scale).
        """
        try:
            score = int(response.split("Rating:")[-1].strip().split()[0])
            return min(max(score, 1), 5)
        except Exception:
            return 3  # Default neutral rating if extraction fails

    def get_human_feedback(self, response):
        """
        Requests human validation feedback for reinforcement learning.
        """
        print("\n🔍 AI Validation Response:")
        print(response)
        while True:
            try:
                rating = int(input("🤖 Please rate this article validation (1-5): "))
                if 1 <= rating <= 5:
                    return rating
                else:
                    print("❌ Invalid input. Enter a number between 1 and 5.")
            except ValueError:
                print("❌ Invalid input. Enter a numeric value.")

    def store_feedback(self, topic, article, ai_rating, human_rating):
        """
        Stores article validation history for RLHF.
        """
        feedback_entry = {
            "topic": topic,
            "article": article,
            "ai_rating": ai_rating,
            "human_rating": human_rating
        }
        self.validation_history.append(feedback_entry)
        if self.verbose:
            print(f"[RLHF] Stored AI Rating: {ai_rating}, Human Rating: {human_rating}")

    def optimize_with_rl(self):
        """
        Reinforcement learning: adjust temperature and max_tokens based on feedback trends.
        """
        if len(self.validation_history) < 5:
            return

        ratings = np.array([entry["human_rating"] for entry in self.validation_history])
        avg_rating = np.mean(ratings)

        if avg_rating < 3:
            self.temperature = max(self.temperature - 0.05, 0.3)
        elif avg_rating > 4:
            self.temperature = min(self.temperature + 0.05, 1.0)

        if any(len(entry["article"]) > self.max_tokens * 0.9 for entry in self.validation_history):
            self.max_tokens = min(self.max_tokens + 50, 1024)

        if self.verbose:
            print(f"[RLHF] Adjusted Settings → Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")

def download_sanitize_report(original_data, sanitized_data, validation_report):
    """
    Creates and enables downloading of a sanitization report.
    Includes original data, sanitized output, and validation notes.
    """
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
"""

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


def download_article_report(original_article, refined_article, validation_report):
    """
    Creates and enables downloading of an article writing/refinement report.
    Includes original article, refined version, and validation.
    """
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
"""

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



if __name__ == "__main__":
    main()
