"""
Context-Aware Response Generation Module.
Synthesizes verified knowledge base articles, detected customer intent,
and emotional sentiment tone into grounded, helpful customer support answers.
Supports Groq API LLM generation with a zero-hallucination deterministic fallback.
"""

import os
from typing import Dict, Any, List, Optional

# Optional Groq client import
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


SYSTEM_PROMPT = """You are an intelligent, empathetic AI Customer Support Assistant.
You must adhere strictly to these operational guidelines:
1. GROUNDING RULE: You may ONLY answer using the verified Knowledge Base FAQ articles provided below.
2. NO HALLUCINATION: If the provided knowledge base does not contain the answer or if the question is out of scope, politely state that you do not have verified documentation for this issue and recommend contacting the human support desk. Do NOT invent policies, timelines, or steps.
3. TONE ADAPTATION: Mirror the designated Customer Sentiment Tone:
   - If Empathetic: sincerely acknowledge the customer's frustration first before providing solutions.
   - If Direct/Professional: be concise and structured.
   - If Warm: be friendly and appreciative.
4. ACTIONABILITY: Include clear, numbered next steps whenever relevant.
5. CONCISENESS: Keep responses focused, professional, and under 150 words.
"""


class ResponseGenerator:
    """
    RAG Response Generator supporting Groq API with robust deterministic fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.temperature = temperature
        self.client = None

        if HAS_GROQ and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def set_api_key(self, api_key: str) -> bool:
        """Dynamically updates the Groq API key and initializes client."""
        self.api_key = api_key.strip() if api_key else None
        if HAS_GROQ and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                return True
            except Exception:
                self.client = None
                return False
        self.client = None
        return False

    def is_llm_active(self) -> bool:
        """Returns True if a live LLM client is configured and ready."""
        return self.client is not None

    def generate(
        self,
        query: str,
        intent_info: Dict[str, Any],
        sentiment_info: Dict[str, Any],
        retrieved_articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates a context-grounded response.
        Uses Groq LLM if active; otherwise generates via grounded rule-based engine.
        """
        # Check relevance of retrieved articles
        has_relevant_doc = bool(retrieved_articles and retrieved_articles[0].get("is_relevant", False))
        top_article = retrieved_articles[0] if retrieved_articles else None

        tone_rec = sentiment_info.get("tone_recommendation", {})
        tone_style = tone_rec.get("style", "Direct, Professional & Efficient")
        empathy_statement = tone_rec.get("empathy_statement", "Thank you for reaching out.")

        # If no relevant article was retrieved from the knowledge base:
        if not has_relevant_doc or top_article is None:
            return self._build_out_of_scope_response(query, empathy_statement, tone_style)

        # Attempt Groq LLM generation if client is active
        if self.is_llm_active():
            try:
                llm_response = self._generate_with_groq(
                    query=query,
                    intent_info=intent_info,
                    sentiment_info=sentiment_info,
                    articles=retrieved_articles
                )
                if llm_response:
                    return {
                        "response": llm_response,
                        "source": "Groq LLM (" + self.model + ")",
                        "is_fallback": False,
                        "matched_faq_id": top_article.get("faq_id"),
                        "matched_faq_title": top_article.get("title"),
                        "similarity_score": top_article.get("similarity_score", 0.0)
                    }
            except Exception as e:
                # Fall back gracefully on API error
                pass

        # Grounded Deterministic Fallback Engine
        fallback_text = self._generate_grounded_fallback(
            query=query,
            top_article=top_article,
            intent_info=intent_info,
            tone_rec=tone_rec
        )

        return {
            "response": fallback_text,
            "source": "Grounded Knowledge Base Engine (Local Fallback)",
            "is_fallback": True,
            "matched_faq_id": top_article.get("faq_id"),
            "matched_faq_title": top_article.get("title"),
            "similarity_score": top_article.get("similarity_score", 0.0)
        }

    def _generate_with_groq(
        self,
        query: str,
        intent_info: Dict[str, Any],
        sentiment_info: Dict[str, Any],
        articles: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Calls Groq Chat Completion API."""
        tone_rec = sentiment_info.get("tone_recommendation", {})

        # Build context from top matching articles
        context_blocks = []
        for i, art in enumerate(articles[:2], start=1):
            block = (
                f"[Article {i}] Title: {art.get('title')}\n"
                f"Question: {art.get('question')}\n"
                f"Answer: {art.get('answer')}\n"
                f"Action Steps: {art.get('action_steps')}\n"
                f"Escalation: {art.get('escalation_contact')}"
            )
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)

        user_prompt = f"""Customer Inquiry: "{query}"

Detected Intent: {intent_info.get('intent')} (Confidence: {intent_info.get('confidence', 0.0):.1%})
Customer Sentiment: {sentiment_info.get('sentiment')} (Tone Required: {tone_rec.get('style')})
Tone Guidance: {tone_rec.get('guidelines')}

VERIFIED KNOWLEDGE BASE CONTEXT:
{context_text}

Please provide a helpful, grounded response that adheres strictly to the verified context and matches the customer's sentiment tone."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()

    def _generate_grounded_fallback(
        self,
        query: str,
        top_article: Dict[str, Any],
        intent_info: Dict[str, Any],
        tone_rec: Dict[str, Any]
    ) -> str:
        """
        Deterministic, zero-hallucination response template strictly composed
        from verified FAQ metadata, adapted to customer emotional state.
        """
        empathy_intro = tone_rec.get("empathy_statement", "Hello!")
        title = top_article.get("title", "Support Article")
        answer = top_article.get("answer", "")
        action_steps = top_article.get("action_steps", "")
        escalation = top_article.get("escalation_contact", "support@example.com")

        response_parts = [
            f"{empathy_intro}\n",
            f"Regarding **{title}**:",
            f"{answer}\n",
            "**Recommended Action Steps:**",
            f"{action_steps}\n",
            f"If you need additional hands-on assistance, please reach our specialized support team directly at `{escalation}`."
        ]

        return "\n".join(response_parts)

    def _build_out_of_scope_response(
        self,
        query: str,
        empathy_statement: str,
        tone_style: str
    ) -> Dict[str, Any]:
        """
        Safe response when no matching documentation exists in the knowledge base.
        Prevents fabrication.
        """
        text = (
            f"{empathy_statement}\n\n"
            "I checked our verified knowledge base documentation, but I could not find a confirmed policy or step-by-step resolution for your specific inquiry.\n\n"
            "To prevent providing inaccurate information, I have flagged this for our specialized support desk. Please contact us directly at `support@example.com` or through live chat during business hours (9 AM - 6 PM EST)."
        )
        return {
            "response": text,
            "source": "Out-of-Scope Guardrail (Grounded Safety)",
            "is_fallback": True,
            "matched_faq_id": None,
            "matched_faq_title": None,
            "similarity_score": 0.0
        }
