
import os
from dotenv import load_dotenv
from google import genai

# Load API key from .env
load_dotenv()

# Create Gemini client
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def generate_answer(query, retrieved_results):

    context = ""

    for i, result in enumerate(retrieved_results, start=1):
        context += f"""
Source {i}:
Document: {result['metadata']['source']}
Page: {result['metadata']['page']}
Content:
{result['text']}
"""

    prompt = f"""
You are SmartGrid AI, an electricity grid advisory assistant.

Answer using only the supplied context.
Explain in simple language.
Do not invent facts.
Mention source document and page number.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        print("Gemini API Error:", str(e))

        return (
            "The ML prediction and document retrieval completed "
            "successfully, but the AI explanation is temporarily "
            "unavailable because the Gemini service could not "
            "respond. Please try again later.\n\n"
            "Retrieved information is listed in the sources below."
        )
