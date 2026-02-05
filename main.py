from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import QueryRequest, FeedbackRequest, update_chat_feedback
from services import get_rag_system

app = FastAPI(title="Cybersecurity Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "running", "message": "Cybersecurity Bot API is active"}

@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        # Lazy load the RAG system only when needed
        rag = get_rag_system()
        return await rag.answer(request)
    except Exception as e:
        # Fallback if system fails to initialize
        return {"answer": "System is initializing or encountered an error. Please try again in a moment.", "context": str(e), "docs": [], "chat_id": ""}

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    update_chat_feedback(request.chat_id, request.feedback)
    return {"status": "success"}

# if __name__ == "__main__":
#     import uvicorn
#     import os
#     print("🚀 Starting Local Server...")
#     uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
