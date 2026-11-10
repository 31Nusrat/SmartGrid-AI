
import os
import json
import time

from dotenv import load_dotenv
from groq import Groq

from ml.prediction import predict_grid_status
from rag.retrieval import retrieve_documents


# ==========================================
# 1. GROQ CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"


# ==========================================
# 2. AGENT TOOL DEFINITIONS
# ==========================================

def get_agent_tools(prediction_available):

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": (
                    "Search the SmartGrid PDF knowledge base "
                    "for information about electricity demand, "
                    "grid stress, outages, weather impacts, "
                    "grid operations and consumer safety."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A focused search query for the "
                                "SmartGrid knowledge base."
                            )
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    if prediction_available:

        tools.append({
            "type": "function",
            "function": {
                "name": "run_prediction",
                "description": (
                    "Run the existing SmartGrid ML models using "
                    "the latest weather and location inputs entered "
                    "by the user in the dashboard. Use this for "
                    "questions asking for predictions, grid risk, "
                    "electricity demand or predicted outage severity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        })

    return tools


# ==========================================
# 3. SEARCH KNOWLEDGE TOOL
# ==========================================

def search_knowledge(query, chunks, embeddings):

    retrieved_results = retrieve_documents(
        query,
        chunks,
        embeddings,
        top_k=3
    )

    if not retrieved_results:
        return {
            "message": "No relevant documents were found.",
            "sources": []
        }

    sources = []

    for item in retrieved_results:

        sources.append({
            "document": item["metadata"]["source"],
            "page": item["metadata"]["page"],
            "similarity_score": round(
                float(item["score"]), 4
            ),
            "content": item["text"]
        })

    return {
        "message": "Relevant documents retrieved successfully.",
        "sources": sources
    }


# ==========================================
# 4. ML PREDICTION TOOL
# ==========================================

def run_prediction_tool(input_data):

    if not input_data:
        return {
            "error": (
                "No prediction inputs are available. "
                "Ask the user to enter the dashboard conditions "
                "and generate a prediction first."
            )
        }

    prediction = predict_grid_status(input_data)

    return prediction


# ==========================================
# 5. MAIN AGENT FUNCTION
# ==========================================

def run_smartgrid_agent(
    question,
    chunks,
    embeddings,
    prediction_input=None,
    chat_history=None
):

    if not question or not question.strip():

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "actions": []
        }

    if chunks is None or embeddings is None:

        return {
            "answer": (
                "The SmartGrid knowledge base is not available. "
                "Please initialize the knowledge base first."
            ),
            "sources": [],
            "actions": []
        }

    start_time = time.time()

    # --------------------------------------
    # Initialize agent conversation
    # --------------------------------------

    system_prompt = """
You are SmartGrid AI Agent, an intelligent electricity
grid advisory assistant.

You have access to two tools:

1. search_knowledge:
   Searches the existing SmartGrid PDF knowledge base.

2. run_prediction:
   Runs the existing ML prediction models using the
   latest inputs entered in the dashboard.

YOUR RESPONSIBILITIES:

- Understand what the user is asking.
- Decide which tools are needed.
- Use RAG for questions about grid concepts,
  outage causes, weather impacts and consumer safety.
- Use ML prediction for requests involving predicted
  electricity demand, grid risk and outage severity.
- For questions requiring both prediction and explanation,
  run the prediction first and then search the PDF documents
  using the prediction results.
- You may call more than one tool when necessary.
- Do not call tools unnecessarily.
- Never invent prediction values or retrieved evidence.
- Clearly distinguish model estimates from confirmed
  real-world grid conditions.
- If prediction inputs are unavailable, explain that
  the user must generate a dashboard prediction first.
- Base factual explanations on retrieved documents.
- Mention source document names and page numbers.
- Give simple, understandable recommendations.

You are an assistant for a prototype system.
Predicted outage severity and affected customers are
model estimates, not confirmed outage events.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # --------------------------------------
    # Add previous conversation
    # --------------------------------------

    if chat_history:

        for message in chat_history[-6:]:

            if message.get("role") in ["user", "assistant"]:

                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

    messages.append({
        "role": "user",
        "content": question
    })

    # --------------------------------------
    # Prepare tools
    # --------------------------------------

    prediction_available = prediction_input is not None

    tools = get_agent_tools(prediction_available)

    actions = []
    all_sources = []
    prediction_result = None

    print("\n========== SMARTGRID AI AGENT ==========")
    print("User question:", question)

    # --------------------------------------
    # Agent decision and tool execution loop
    # --------------------------------------

    max_iterations = 5

    try:

        for iteration in range(max_iterations):

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_completion_tokens=2048
            )

            assistant_message = response.choices[0].message

            # ----------------------------------
            # No more tools: final response
            # ----------------------------------

            if not assistant_message.tool_calls:

                answer = assistant_message.content or (
                    "I couldn't generate a response. Please try again."
                )

                print("\nAgent completed successfully!")
                print(
                    "Total agent time:",
                    round(time.time() - start_time, 2),
                    "seconds"
                )

                return {
                    "answer": answer,
                    "sources": all_sources,
                    "actions": actions,
                    "prediction": prediction_result
                }

            # ----------------------------------
            # Save assistant tool-call message
            # ----------------------------------

            messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            # ----------------------------------
            # Execute requested tools
            # ----------------------------------

            for tool_call in assistant_message.tool_calls:

                function_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    print("\nAgent selected tool:", function_name)

                    # SEARCH KNOWLEDGE
                    if function_name == "search_knowledge":

                        query = arguments.get("query", "").strip()

                        if not query:
                            raise ValueError(
                                "Search query cannot be empty."
                            )

                        tool_result = search_knowledge(
                            query,
                            chunks,
                            embeddings
                        )

                        all_sources.extend(
                            tool_result.get("sources", [])
                        )

                        actions.append("Searched PDF knowledge base")

                    # RUN ML PREDICTION
                    elif function_name == "run_prediction":

                        if not prediction_available:
                            raise ValueError(
                                "Prediction inputs are unavailable."
                            )

                        prediction_result = run_prediction_tool(
                            prediction_input
                        )

                        tool_result = prediction_result

                        actions.append("Executed ML prediction models")

                    else:

                        raise ValueError(
                            f"Unknown tool: {function_name}"
                        )

                except Exception as tool_error:

                    tool_result = {
                        "error": str(tool_error)
                    }

                # ----------------------------------
                # Send tool result back to Groq
                # ----------------------------------

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str
                    )
                })

        return {
            "answer": (
                "The agent reached its maximum number of tool "
                "execution steps. Please try a simpler question."
            ),
            "sources": all_sources,
            "actions": actions,
            "prediction": prediction_result
        }

    except Exception as e:

        print("SmartGrid Agent Error:", repr(e))

        return {
            "answer": (
                "The SmartGrid AI Agent is temporarily unavailable. "
                "Please try again later."
            ),
            "sources": all_sources,
            "actions": actions,
            "prediction": prediction_result
        }
