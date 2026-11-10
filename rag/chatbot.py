
from rag.smartgrid_pipeline import load_knowledge_base
from rag.retrieval import retrieve_documents
from rag.llm import generate_answer


# ==========================================
# 1. INITIALIZE CHATBOT
# ==========================================

def initialize_chatbot():

    print("Initializing SmartGrid AI Chatbot...")

    chunks, embeddings = load_knowledge_base()

    print("Chatbot knowledge base is ready!")

    return chunks, embeddings


# ==========================================
# 2. CHAT FUNCTION
# ==========================================

def ask_chatbot(question, chunks, embeddings, chat_history):

    if not question or not question.strip():
        return "Please enter a question.", []

    # Retrieve relevant PDF content
    retrieved_results = retrieve_documents(
        question,
        chunks,
        embeddings,
        top_k=3
    )

    if not retrieved_results:
        return (
            "I couldn't find relevant information in the "
            "available documents. Please try another question.",
            []
        )

    # Generate answer using Gemini and conversation history
    answer = generate_answer(
        question,
        retrieved_results,
        chat_history
    )

    # Prepare PDF source information
    sources = [
        {
            "document": item["metadata"]["source"],
            "page": item["metadata"]["page"],
            "similarity_score": round(float(item["score"]), 4)
        }
        for item in retrieved_results
    ]

    # Update conversation history
    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, sources


# ==========================================
# 3. TERMINAL CHATBOT
# ==========================================

if __name__ == "__main__":

    try:

        chunks, embeddings = initialize_chatbot()

        conversation_history = []

        print("\n" + "=" * 55)
        print("          SMARTGRID AI CHATBOT")
        print("=" * 55)

        print("Ask questions about electricity and power grids.")
        print("Type 'exit' to stop the chatbot.")

        while True:

            question = input("\nYou: ").strip()

            if question.lower() in ["exit", "quit", "bye"]:
                print("\nSmartGrid AI: Goodbye!")
                break

            answer, sources = ask_chatbot(
                question,
                chunks,
                embeddings,
                conversation_history
            )

            print("\nSmartGrid AI:")
            print(answer)

            if sources:

                print("\nRetrieved Sources:")

                for index, source in enumerate(sources, start=1):

                    print(f"\nSource {index}")
                    print("Document:", source["document"])
                    print("Page:", source["page"])
                    print("Similarity:", source["similarity_score"])

    except Exception as e:

        print("\nCHATBOT ERROR")
        print(type(e).__name__ + ":", str(e))
