"""
Streamlit Web Interface for AI Customer Support Assistant.
Provides a modern chatbot interface with real-time intent classification,
sentiment analysis, FAISS knowledge base retrieval, and grounded response generation.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

# Configure project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.intent_classifier import IntentClassifier
from src.retriever import FaissRetriever
from src.sentiment_analyzer import SentimentAnalyzer
from src.response_generator import ResponseGenerator
from src.utils import SupportAssistantPipeline

# Page configuration
st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished appearance
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .badge-intent {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .badge-sentiment-neg {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .badge-sentiment-pos {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .badge-sentiment-neu {
        background-color: #f3f4f6;
        color: #374151;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .badge-source {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="Loading trained ML models and FAISS vector index...")
def load_pipeline():
    """Caches pipeline initialization so it only loads once per session."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    clf_path = os.path.join(models_dir, "intent_classifier.joblib")

    if not os.path.exists(clf_path):
        st.error("Trained models not found! Run 'python scripts/train.py' first.")
        st.stop()

    classifier = IntentClassifier(clf_path)
    retriever = FaissRetriever().load(models_dir)
    sentiment_analyzer = SentimentAnalyzer()
    generator = ResponseGenerator()

    return SupportAssistantPipeline(
        classifier=classifier,
        retriever=retriever,
        sentiment_analyzer=sentiment_analyzer,
        generator=generator
    )


pipeline = load_pipeline()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am your AI Customer Support Assistant. How can I assist you with your orders, payments, refunds, or technical inquiries today?",
            "metadata": None
        }
    ]

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/help.png", width=110)
    st.title("Settings & Controls")

    st.subheader("LLM Configuration")
    groq_key = st.text_input(
        "Groq API Key (Optional)",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Leave blank to run in 100% offline, zero-hallucination Grounded Fallback Mode."
    )
    if groq_key:
        active = pipeline.generator.set_api_key(groq_key)
        if active:
            st.success("🟢 Groq API Connected")
        else:
            st.warning("⚠️ Invalid Groq Key, running in Local Mode")
    else:
        pipeline.generator.set_api_key("")
        st.info("ℹ️ Running in Grounded Local Fallback Mode (No API key needed)")

    selected_model = st.selectbox(
        "Groq LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    pipeline.generator.model = selected_model

    st.markdown("---")
    st.subheader("⚡ Quick Sample Inquiries")
    sample_queries = [
        "Where is my refund for order #84910?",
        "My credit card was declined at checkout with an error.",
        "The mobile app freezes and crashes whenever I view cart.",
        "Please cancel my order #9921 before it ships!",
        "I forgot my password and cannot log in."
    ]

    selected_sample = None
    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            selected_sample = sq

    st.markdown("---")
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Conversation reset! How can I help you today?",
                "metadata": None
            }
        ]
        st.rerun()


# Main Tabs: Chat Assistant vs Evaluation Dashboard vs Knowledge Base
tab_chat, tab_metrics, tab_kb = st.tabs(["💬 Support Chatbot", "📊 Model Evaluation & Metrics", "📚 Knowledge Base"])

with tab_chat:
    st.markdown("<div class='main-header'>AI Customer Support Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Powered by TF-IDF Intent Classification, VADER Sentiment Tone Modulation, and FAISS Vector Retrieval.</div>",
        unsafe_allow_html=True
    )

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # Render Metadata badges if available
            meta = msg.get("metadata")
            if meta:
                intent_val = meta.get("intent", {}).get("intent", "Unknown")
                conf_val = meta.get("intent", {}).get("confidence", 0.0)
                sentiment_val = meta.get("sentiment", {}).get("sentiment", "neutral")
                tone_style = meta.get("sentiment", {}).get("tone_recommendation", {}).get("style", "")
                src_val = meta.get("generation", {}).get("source", "Knowledge Base")

                badge_class = "badge-sentiment-neu"
                if sentiment_val == "positive":
                    badge_class = "badge-sentiment-pos"
                elif sentiment_val == "negative":
                    badge_class = "badge-sentiment-neg"

                st.markdown(
                    f"""
                    <div>
                        <span class='badge-intent'>🎯 Intent: {intent_val} ({conf_val:.1%})</span>
                        <span class='{badge_class}'>💬 Sentiment: {sentiment_val.capitalize()}</span>
                        <span class='badge-source'>⚙️ {src_val}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Expandable retrieval details
                with st.expander("🔍 Inspect Retrieved FAQ Source Document"):
                    top_faq = meta.get("generation", {})
                    st.write(f"**Matched FAQ:** {top_faq.get('matched_faq_title')} (`{top_faq.get('matched_faq_id')}`)")
                    st.write(f"**Semantic Similarity Score:** `{top_faq.get('similarity_score', 0.0):.4f}`")
                    if meta.get("retrieved_articles"):
                        art = meta["retrieved_articles"][0]
                        st.markdown(f"**Full Answer:** {art.get('answer')}")
                        st.markdown(f"**Action Steps:** {art.get('action_steps')}")
                        st.markdown(f"**Escalation Desk:** `{art.get('escalation_contact')}`")

    # Handle Input
    user_query = st.chat_input("Type your customer service inquiry here...")
    if selected_sample:
        user_query = selected_sample

    if user_query:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query, "metadata": None})
        with st.chat_message("user"):
            st.write(user_query)

        # Process Query through Pipeline
        with st.spinner("Processing intent, sentiment, and searching knowledge base..."):
            result = pipeline.process_query(user_query)

        # Append Assistant Response
        assistant_reply = result["generation"]["response"]
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_reply,
            "metadata": result
        })

        st.rerun()


with tab_metrics:
    st.header("📈 Model Evaluation & Performance")
    st.write("Held-out test set evaluation across 6 intent categories (80% train / 20% test stratified split).")

    metrics_file = os.path.join(PROJECT_ROOT, "models", "evaluation_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", f"{metrics_data.get('accuracy', 0.0):.2%}")
        col2.metric("Weighted Precision", f"{metrics_data.get('precision_weighted', 0.0):.2%}")
        col3.metric("Weighted Recall", f"{metrics_data.get('recall_weighted', 0.0):.2%}")
        col4.metric("Weighted F1-Score", f"{metrics_data.get('f1_weighted', 0.0):.2%}")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Confusion Matrix")
            cm_img = os.path.join(PROJECT_ROOT, "models", "confusion_matrix.png")
            if os.path.exists(cm_img):
                st.image(cm_img, use_container_width=True)

        with col_right:
            st.subheader("Metric Summary")
            bar_img = os.path.join(PROJECT_ROOT, "models", "metrics_summary.png")
            if os.path.exists(bar_img):
                st.image(bar_img, use_container_width=True)

        st.subheader("Detailed Classification Report by Intent Class")
        rep = metrics_data.get("classification_report", {})
        if rep:
            df_rep = pd.DataFrame(rep).transpose()
            st.dataframe(df_rep.style.format("{:.2%}"), use_container_width=True)
    else:
        st.info("Metrics not generated yet. Run scripts/train.py to generate evaluation metrics.")


with tab_kb:
    st.header("📚 Customer Support Knowledge Base")
    st.write("Indexed documents used by FAISS vector retriever for zero-hallucination grounding.")

    kb_path = os.path.join(PROJECT_ROOT, "data", "knowledge_base.csv")
    if os.path.exists(kb_path):
        df_kb = pd.read_csv(kb_path)
        search_filter = st.text_input("Filter articles by keyword or category:")
        if search_filter:
            filtered = df_kb[
                df_kb["title"].str.contains(search_filter, case=False, na=False) |
                df_kb["category"].str.contains(search_filter, case=False, na=False) |
                df_kb["keywords"].str.contains(search_filter, case=False, na=False)
            ]
        else:
            filtered = df_kb

        st.write(f"Showing **{len(filtered)}** articles:")
        for _, row in filtered.iterrows():
            with st.expander(f"[{row['category'].upper()}] {row['title']} ({row['faq_id']})"):
                st.markdown(f"**Customer Question:** {row['question']}")
                st.markdown(f"**Official Answer:** {row['answer']}")
                st.markdown(f"**Action Steps:** {row['action_steps']}")
                st.markdown(f"**Keywords:** `{row['keywords']}`")
                st.markdown(f"**Escalation Desk:** `{row['escalation_contact']}`")
    else:
        st.error("Knowledge base CSV not found in data/.")
