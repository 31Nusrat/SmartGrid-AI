
import os
import time

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# 1. LOAD GROQ API CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )

client = Groq(api_key=api_key)


# ==========================================
# 2. GENERATE RAG ANSWER
# ==========================================

def generate_answer(query, retrieved_results, chat_history=None):

    total_start = time.time()

    # --------------------------------------
    # Prepare retrieved document context
    # --------------------------------------

    context = ""

    for i, result in enumerate(retrieved_results, start=1):

        context += f"""
Source {i}:
Document: {result['metadata']['source']}
Page: {result['metadata']['page']}

Content:
{result['text']}
"""

    # --------------------------------------
    # Prepare conversation history
    # --------------------------------------

    history_text = ""

    if chat_history:

        for message in chat_history[-6:]:

            history_text += (
                f"{message['role'].upper()}: "
                f"{message['content']}\n"
            )

    # --------------------------------------
    # Construct RAG prompt
    # --------------------------------------

    prompt = f"""
You are SmartGrid AI, an electricity grid advisory assistant.

Your responsibilities:
- Answer questions related to electricity, power grids,
  outages, weather impacts and consumer safety.
- Use the supplied document context as your factual source.
- Explain information in simple and understandable language.
- Do not invent facts, statistics or recommendations.
- Use conversation history to understand follow-up questions.
- If the supplied documents do not contain the answer,
  clearly say that the information is unavailable.
- Mention source document names and page numbers where relevant.

PREVIOUS CONVERSATION:
{history_text}

RETRIEVED DOCUMENT CONTEXT:
{context}

CURRENT USER QUESTION:
{query}

ANSWER:
"""

    print(
        "Prompt preparation:",
        round(time.time() - total_start, 2),
        "seconds"
    )

    # --------------------------------------
    # Generate response using Groq
    # --------------------------------------

    api_start = time.time()

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_completion_tokens=2048
        )

        print(
            "Groq API time:",
            round(time.time() - api_start, 2),
            "seconds"
        )

        answer = response.choices[0].message.content

        if answer:
            return answer

        return "Groq returned an empty response. Please try again."

    except Exception as e:

        print("Groq API Error:", repr(e))

        return (
            "The AI explanation is temporarily unavailable "
            "because the Groq service could not respond. "
            "Please try again later."
        )

    finally:

        print(
            "Total LLM time:",
            round(time.time() - total_start, 2),
            "seconds"
        )
