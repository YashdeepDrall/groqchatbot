print("APP FILE LOADED")

import difflib
#import uvicorn
import os
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

# LangChain and Vector Store imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Local imports from other files
from database import store_chat, update_chat_feedback, QueryRequest, FeedbackRequest
from analysis import analyze_response
from create_vectorstore import create_vector_store_from_directory

# --- Initial Setup ---
load_dotenv()

# --- LangChain/Model Setup ---
# Initialize globals to None to prevent startup crashes
db = None
model = None
embeddings = None

try:
    # --- API Key Validation ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("⚠️ WARNING: GROQ_API_KEY not found. The bot will not function correctly.")

    print("Loading models and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Always rebuild vector store on startup to ensure latest data is used
    print("🔄 Rebuilding vector store from 'data' directory...")
    create_vector_store_from_directory()

    if os.path.exists("vectorstore"):
        db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded.")
    
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    print("✅ Models loaded.")
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    print(traceback.format_exc())

# --- Prompt Templates ---
condense_question_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert at understanding user intent in a conversation.
    Your task is to rewrite the user's follow-up input into a clear, standalone question that can be used to search a knowledge base.

    **CRITICAL RULE FOR AFFIRMATIONS:**
    If the user says "yes", "sure", "okay", or similar, look at the Consultant's last message.
    The Consultant likely offered information (e.g., "Would you like to know about X?").
    You MUST rewrite "yes" into "Tell me about X." or "What is X?".
    
    **Examples:**
    
    Chat History:
    Consultant: IIRIS is a global firm. Would you like to know about our services?
    User: yes
    Standalone Question: What services does IIRIS offer?

    Chat History:
    Consultant: Would you like to know more about the services IIRIS offers or its areas of expertise?
    User: yes
    Standalone Question: What are the services and areas of expertise of IIRIS?

    **Now process this:**
    
    Chat History:
    {history}

    Follow Up Input: {question}

    Standalone Question:
    """
)

context_prompt = ChatPromptTemplate.from_template(
    """
    You are a professional consultant at IIRIS Consulting.
    Your primary goal is to answer the client's question using the provided Context.

    Rules:
    - Base your answer strictly on the provided Context.
    - If the Context provides a direct answer, use it.
    - If the Context does not contain a direct answer but has related information, use that to provide a helpful response. For example, if asked for a CEO and the context lists VPs, state that the CEO is not listed and provide the VP information from the context.
    - If the Context is completely irrelevant to the question and does not help in answering it at all, then and only then should you state that you do not have enough information.
    - Be professional and practical.
    - Do NOT guess or hallucinate.
    - Use the provided Chat History to understand the conversation context.
    - Do NOT mention internal document identifiers (e.g., "Q14", "Context", "Document"). Provide the answer naturally.

    Chat History:
    {history}

    Context:
    {context}

    Client Question:
    {question}

    Consultant Answer:
    """
)

relevance_prompt = ChatPromptTemplate.from_template(
    "You are a classification system. Your task is to determine if a user's question is relevant to IIRIS Consulting. "
    "Relevant topics include the company's services, operations, global presence, leadership, team members (e.g., CEO, VP, AVP), "
    "company sub-brands (like IntelliRisk), cybersecurity, risk management, and brand protection. "
    "Answer only with YES or NO.\n\nQuestion: {question}\n\nIs this question relevant? Answer:"
)

internal_knowledge_prompt = ChatPromptTemplate.from_template(
    "You are a professional cybersecurity consultant at IIRIS Consulting.\n"
    "The user has asked a question that is not covered by your current context documents.\n"
    "Answer the question using your internal knowledge. Be professional and practical.\n"
    "If you do not know the answer, simply state that you do not have that specific information at the moment. Do NOT use phrases like 'database not up-to-date' or 'information not publicly available'.\n"
    "Do NOT guess or hallucinate.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# --- FastAPI Application ---
app = FastAPI(title="Cybersecurity Bot API")

# Add a root endpoint for Render Health Checks
@app.get("/")
def read_root():
    return {"status": "running", "message": "Cybersecurity Bot API is active"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def is_greeting(question: str) -> bool:
    greeting_words = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    cleaned_input = question.lower().strip("!.,? '\"")
    if cleaned_input in greeting_words or any(cleaned_input.startswith(g + " ") for g in greeting_words):
        return True
    return bool(difflib.get_close_matches(cleaned_input, greeting_words, n=1, cutoff=0.8))

def is_signoff(question: str) -> bool:
    signoff_words = ["bye", "goodbye", "see you", "thank you", "thanks", "thankyou", "exit", "quit"]
    cleaned_input = question.lower().strip("!.,? '\"")
    if cleaned_input in signoff_words or any(cleaned_input.startswith(s + " ") for s in signoff_words):
        return True
    return bool(difflib.get_close_matches(cleaned_input, signoff_words, n=1, cutoff=0.8))

def is_affirmation(question: str) -> bool:
    """Checks if the input is a simple affirmation like 'yes' or 'sure'."""
    affirmation_words = ["yes", "yep", "yeah", "sure", "ok", "okay", "go ahead", "please do"]
    cleaned_input = question.lower().strip("!.,? '\"")
    return cleaned_input in affirmation_words

def correct_typos(question: str) -> str:
    """Corrects typos in the user's question by matching against a list of known domain terms."""
    known_terms = ["IIRIS", "IntelliRisk", "IntelliFuture", "IIRIS Knowledge", "Consulting", "Cybersecurity", "Forensics"]
    words = question.split()
    corrected_words = []
    
    for word in words:
        clean_word = word.strip("!.,? '\"").lower()
        if not clean_word:
            corrected_words.append(word)
            continue
            
        matches = difflib.get_close_matches(clean_word, [t.lower() for t in known_terms], n=1, cutoff=0.8)
        if matches:
            match = matches[0]
            correct_term = next(t for t in known_terms if t.lower() == match)
            # Preserve punctuation
            lower_word = word.lower()
            start_index = lower_word.find(clean_word)
            if start_index != -1:
                prefix = word[:start_index]
                suffix = word[start_index + len(clean_word):]
                corrected_words.append(prefix + correct_term + suffix)
            else:
                corrected_words.append(correct_term)
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)

# --- Diagnostic Endpoint ---
@app.get("/test-groq")
async def test_groq_endpoint():
    """A simple endpoint to test the connection to the Groq API."""
    try:
        print("\n🧪 Testing Groq API connection...")
        test_prompt = ChatPromptTemplate.from_template("Why is the sky blue? Answer in one sentence.")
        chain = test_prompt | model
        response = await chain.ainvoke({})
        print(f"✅ Groq API test successful. Response: {response.content}")
        return {"status": "success", "response": response.content}
    except Exception as e:
        print(f"❌ Groq API test failed: {e}\n{traceback.format_exc()}")
        return {"status": "failed", "error": str(e)}

# --- API Endpoints ---
@app.post("/ask")
async def ask_endpoint(request: QueryRequest):
    # Check if system is ready
    if db is None or model is None:
        return {"answer": "System is currently initializing or encountered an error loading the knowledge base. Please check server logs.", "context": "System Error", "docs": [], "chat_id": ""}

    def format_history(history: List[Dict[str, str]]) -> str:
        formatted = ""
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Consultant"
            formatted += f"{role}: {msg.get('content')}\n"
        return formatted

    try:
        # Initialize usage stats
        total_usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        def update_usage(response_obj):
            if hasattr(response_obj, 'response_metadata'):
                usage = response_obj.response_metadata.get('token_usage', {})
                if usage:
                    total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_usage["total_tokens"] += usage.get("total_tokens", 0)

        # Pre-process question to handle typos
        corrected_question = correct_typos(request.question)

        if is_greeting(corrected_question):
            # Only treat as a pure greeting if the sentence is short (<= 3 words).
            # If longer, it likely contains a question (e.g., "Hi, what is IIRIS?"), so we fall through to RAG.
            if len(corrected_question.split()) <= 3:
                response = "Hello! How may I help you today?"
                chat_id = store_chat(request.question, response, "", flags=["greeting"], token_usage=total_usage)
                return {"answer": response, "context": "", "docs": [], "chat_id": chat_id, "usage": total_usage}

        if is_signoff(corrected_question):
            response = "Happy to help you! I am here for any assistance. Is there anything else I can help you with ?"
            chat_id = store_chat(request.question, response, "", flags=["signoff"], token_usage=total_usage)
            return {"answer": response, "context": "", "docs": [], "chat_id": chat_id, "usage": total_usage}

        # Create a model instance with request-specific parameters
        request_model = model.bind(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Process History and Contextualize Question
        history_text = format_history(request.history)
        search_query = corrected_question
        
        if history_text:
            # 1. Check for simple affirmation FIRST (e.g., "yes", "sure")
            # This bypasses the generic condensation model which often fails on short affirmations.
            if is_affirmation(corrected_question):
                print("🔍 User provided affirmation. Attempting to resolve implied question from history.")
                last_bot_message = ""
                for msg in reversed(request.history):
                    if msg.get("role") != "user":
                        last_bot_message = msg.get("content")
                        break
                
                if last_bot_message:
                    affirmation_prompt = ChatPromptTemplate.from_template(
                        """
                        The user replied with a simple affirmation (e.g., "yes") to the assistant's last message.
                        
                        Assistant's last message: "{last_question}"
                        
                        Task: Interpret the affirmation and formulate the specific question the user is implying.
                        
                        Examples:
                        - Assistant: "Would you like to know about our services?" -> User: "Yes" -> Implied Question: "What services does IIRIS offer?"
                        - Assistant: "Do you want to know our location?" -> User: "Sure" -> Implied Question: "Where is IIRIS located?"
                        
                        Implied Question:
                        """
                    )
                    affirmation_chain = affirmation_prompt | request_model
                    affirmation_response = await affirmation_chain.ainvoke({"last_question": last_bot_message})
                    update_usage(affirmation_response)
                    search_query = affirmation_response.content.strip()
                    print(f"✅ Resolved affirmation to: '{search_query}'")
            else:
                # 2. Standard condensation for other follow-ups
                condensed_response = await (condense_question_prompt | request_model).ainvoke({
                    "history": history_text, 
                    "question": corrected_question
                })
                update_usage(condensed_response)
                search_query = condensed_response.content.strip()

        # Use asynchronous search for better performance
        docs = await db.asimilarity_search(search_query, k=request.k)
        context = "\n".join(d.page_content for d in docs)

        # Always try to answer from context first using the detailed prompt.
        response_obj = await (context_prompt | request_model).ainvoke({
            "context": context, 
            "question": search_query, # Use the rephrased question for answering
            "history": history_text
        })
        update_usage(response_obj)
        response = response_obj.content

        # Analyze the response to see if it was a good answer.
        flags = analyze_response(corrected_question, response)

        # If the model couldn't answer from context, check relevance and try internal knowledge.
        if "unable_to_answer" in flags:
            relevance_chain = relevance_prompt | request_model
            relevance_response = await relevance_chain.ainvoke({"question": search_query})
            update_usage(relevance_response)
            is_relevant = "yes" in relevance_response.content.lower()

            if is_relevant:
                # The question is relevant, but context was not enough. Use internal knowledge.
                response_obj = await (internal_knowledge_prompt | request_model).ainvoke({"question": search_query})
                update_usage(response_obj)
                response = response_obj.content
                context = "Internal Knowledge Used"
                docs = [] # Clear docs since we are using internal knowledge
                # Re-analyze the new response, keeping only the new flags.
                flags = analyze_response(corrected_question, response)
            else:
                # The question is not relevant to our domain.
                response = "I specialize in cybersecurity consulting and IIRIS services. Please ask a relevant question."
                context, docs = "", []
                flags.append("irrelevant_question")

        if "irrelevant_question" not in flags and "greeting" not in flags and "signoff" not in flags:
            # If any flags were raised during the process (e.g., "unable_to_answer"), append the support contact.
            # The redundant call to analyze_response is removed as it was causing incorrect flagging.
            if list(set(flags)):
                response += "\n\nFor further assistance, please contact support at: contactus@iirisconsulting.com"
        
        chat_id = store_chat(request.question, response, context, flags=flags, token_usage=total_usage)
        serialized_docs = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        return {"answer": response, "context": context, "docs": serialized_docs, "chat_id": chat_id, "usage": total_usage}

    except Exception as e:
        # Print the full error traceback for detailed debugging
        print(f"❌ An error occurred in /ask endpoint: {e}\n{traceback.format_exc()}")
        usage_to_log = locals().get("total_usage", {})
        chat_id = store_chat(request.question, "Sorry, I encountered an error. Please try again.", "Error", flags=["error"], token_usage=usage_to_log)
        return {"answer": "Sorry, I encountered an error. Please try again.", "context": "Error", "docs": [], "chat_id": chat_id}

@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest):
    update_chat_feedback(request.chat_id, request.feedback)
    return {"status": "success"}

# if __name__ == "__main__":
#     print("🚀 Starting Backend Server...")
#     # Read host and port from environment variables, with defaults
#     # This is the correct setup for cloud services like Render.
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", 10000))
#     print(f"✅ Attempting to start server on host='{host}' and port='{port}' from environment variables.")
#     uvicorn.run(app, host=host, port=port)
