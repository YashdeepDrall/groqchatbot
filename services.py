import os
import traceback
import difflib
from typing import List, Dict
from dotenv import load_dotenv

# Heavy imports moved here
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from database import store_chat, QueryRequest
from analysis import analyze_response
from create_vectorstore import create_vector_store_from_directory

load_dotenv()

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
    "Gurpawan Singh and Garry Singh are the same guys, So if the context mentions either of them, you can use that information to answer questions about the CEO or leadership.\n"
    "Answer the question using your internal knowledge. Be professional and practical.\n"
    "If you do not know the answer, simply state that you do not have that specific information at the moment. Do NOT use phrases like 'database not up-to-date' or 'information not publicly available'.\n"
    "Do NOT guess or hallucinate.\n\n"
    "Question: {question}\n\n"
    "Answer:"
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
    affirmation_words = ["yes", "yep", "yeah", "sure", "ok", "okay", "go ahead", "please do"]
    cleaned_input = question.lower().strip("!.,? '\"")
    return cleaned_input in affirmation_words

def correct_typos(question: str) -> str:
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

# --- RAG System Class ---
class RagSystem:
    def __init__(self):
        # Lazy imports to prevent slow startup
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq

        print("🔄 Initializing RAG system (Lazy Load)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Check/Create vectorstore
        if not os.path.exists("vectorstore"):
            print("⚠️ Vector store not found. Creating it now...")
            create_vector_store_from_directory()
        else:
            pass

        self.db = FAISS.load_local("vectorstore", self.embeddings, allow_dangerous_deserialization=True)
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("⚠️ WARNING: GROQ_API_KEY not found.")
        
        self.model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=groq_api_key)
        print("✅ RAG System Ready.")

    async def answer(self, request: QueryRequest):
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

            def format_history(history: List[Dict[str, str]]) -> str:
                formatted = ""
                for msg in history:
                    role = "User" if msg.get("role") == "user" else "Consultant"
                    formatted += f"{role}: {msg.get('content')}\n"
                return formatted

            corrected_question = correct_typos(request.question)

            # Greeting Check
            if is_greeting(corrected_question):
                if len(corrected_question.split()) <= 3:
                    response = "Hello! How may I help you today?"
                    chat_id = store_chat(request.question, response, "", flags=["greeting"], token_usage=total_usage)
                    return {"answer": response, "context": "", "docs": [], "chat_id": chat_id, "usage": total_usage}

            # Signoff Check
            if is_signoff(corrected_question):
                response = "Happy to help you! I am here for any assistance. Is there anything else I can help you with ?"
                chat_id = store_chat(request.question, response, "", flags=["signoff"], token_usage=total_usage)
                return {"answer": response, "context": "", "docs": [], "chat_id": chat_id, "usage": total_usage}

            request_model = self.model.bind(temperature=request.temperature, max_tokens=request.max_tokens)
            history_text = format_history(request.history)
            search_query = corrected_question
            
            # Contextualize Question
            if history_text:
                if is_affirmation(corrected_question):
                    last_bot_message = ""
                    for msg in reversed(request.history):
                        if msg.get("role") != "user":
                            last_bot_message = msg.get("content")
                            break
                    if last_bot_message:
                        affirmation_prompt = ChatPromptTemplate.from_template(
                            """The user replied with a simple affirmation (e.g., "yes") to the assistant's last message.
                            Assistant's last message: "{last_question}"
                            Task: Interpret the affirmation and formulate the specific question the user is implying.
                            Implied Question:"""
                        )
                        affirmation_chain = affirmation_prompt | request_model
                        affirmation_response = await affirmation_chain.ainvoke({"last_question": last_bot_message})
                        update_usage(affirmation_response)
                        search_query = affirmation_response.content.strip()
                else:
                    condensed_response = await (condense_question_prompt | request_model).ainvoke({
                        "history": history_text, 
                        "question": corrected_question
                    })
                    update_usage(condensed_response)
                    search_query = condensed_response.content.strip()

            # Search
            docs = await self.db.asimilarity_search(search_query, k=request.k)
            context = "\n".join(d.page_content for d in docs)

            # Answer
            response_obj = await (context_prompt | request_model).ainvoke({
                "context": context, 
                "question": search_query,
                "history": history_text
            })
            update_usage(response_obj)
            response = response_obj.content

            # Analyze & Fallback
            flags = analyze_response(corrected_question, response)

            if "unable_to_answer" in flags:
                relevance_chain = relevance_prompt | request_model
                relevance_response = await relevance_chain.ainvoke({"question": search_query})
                update_usage(relevance_response)
                is_relevant = "yes" in relevance_response.content.lower()

                if is_relevant:
                    response_obj = await (internal_knowledge_prompt | request_model).ainvoke({"question": search_query})
                    update_usage(response_obj)
                    response = response_obj.content
                    context = "Internal Knowledge Used"
                    docs = []
                    flags = analyze_response(corrected_question, response)
                else:
                    response = "I specialize in cybersecurity consulting and IIRIS services. Please ask a relevant question."
                    context, docs = "", []
                    flags.append("irrelevant_question")

            if "irrelevant_question" not in flags and "greeting" not in flags and "signoff" not in flags:
                if list(set(flags)):
                    response += "\n\nFor further assistance, please contact support at: contactus@iirisconsulting.com"
            
            chat_id = store_chat(request.question, response, context, flags=flags, token_usage=total_usage)
            serialized_docs = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
            return {"answer": response, "context": context, "docs": serialized_docs, "chat_id": chat_id, "usage": total_usage}

        except Exception as e:
            print(f"❌ Error in RAG handler: {e}\n{traceback.format_exc()}")
            raise e

# --- Singleton Pattern ---
_rag_system = None

def get_rag_system():
    global _rag_system
    if _rag_system is None:
        _rag_system = RagSystem()
    return _rag_system