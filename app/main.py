from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import traceback

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv()  # For local development only

# -----------------------------
# 2️⃣ Setup Python path for book_data
# -----------------------------
# backend_root should contain book_data.py and app/
backend_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(backend_root))

try:
    from book_data import get_all_chapters
    print("✅ Successfully imported book_data")
except ModuleNotFoundError as e:
    print(f"❌ Error importing book_data: {e}")
    print("Make sure book_data.py is in the backend folder (same level as app/)")
    raise

# -----------------------------
# 3️⃣ Configure FastAPI
# -----------------------------
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Add frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 4️⃣ Configure Gemini API
# -----------------------------
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️  WARNING: GEMINI_API_KEY not found in environment!")
else:
    print(f"✅ Gemini API Key loaded: {api_key[:10]}...")
    genai.configure(api_key=api_key)

# -----------------------------
# 5️⃣ Define Models
# -----------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    source: str
    chapters: List[int] = []

# -----------------------------
# 6️⃣ Utility: Search book content
# -----------------------------
def search_book_content(query: str, top_k: int = 3):
    try:
        chapters_data = get_all_chapters()
        query_lower = query.lower()
        query_words = [word for word in query_lower.split() if len(word) > 2]
        
        results = []
        for chapter in chapters_data:
            title_lower = chapter['title'].lower()
            content_lower = chapter['content'].lower()
            
            score = 0
            for word in query_words:
                score += title_lower.count(word) * 3
                score += content_lower.count(word)
            
            if score > 0:
                results.append({'chapter': chapter, 'score': score})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:top_k]
        total_score = sum(r['score'] for r in top_results)
        
        return [r['chapter'] for r in top_results], total_score
    except Exception as e:
        print(f"❌ Error in search_book_content: {e}")
        traceback.print_exc()
        raise

# -----------------------------
# 7️⃣ Chat Endpoint
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        relevant_chapters, relevance_score = search_book_content(request.message, top_k=3)
        RELEVANCE_THRESHOLD = 3

        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="Gemini API key not configured. Check your environment variables."
            )

        model = genai.GenerativeModel('gemini-2.5-flash')

        if relevance_score >= RELEVANCE_THRESHOLD and relevant_chapters:
            context = "\n\n".join([
                f"=== CHAPTER {ch['id']}: {ch['title']} ===\n"
                f"Difficulty: {ch['difficulty']} | Reading Time: {ch['readingTime']} | Pages: {ch['pages']}\n\n"
                f"{ch['content']}"
                for ch in relevant_chapters
            ])
            
            chapter_numbers = [ch['id'] for ch in relevant_chapters]
            
            prompt = f"""You are an AI assistant for "Physical AI and Humanoid Robotics" by Azmat Ali.

📚 INSTRUCTIONS:
- Answer ONLY from the book chapters below
- ALWAYS cite chapter numbers: "According to Chapter X..." or "(Chapter Y, pages Z)"
- If exact info isn't in chapters, say: "The provided chapters discuss [related topic] but not specifically about [query]"
- Be clear, concise, and accurate
- Format citations as: Chapter X (pages Y-Z)

📖 BOOK CHAPTERS:
{context}

❓ USER QUESTION: {request.message}

💡 YOUR ANSWER (with citations):"""

            response = model.generate_content(prompt)
            
            return ChatResponse(
                response=response.text,
                source="book",
                chapters=chapter_numbers
            )
        else:
            prompt = f"""You are helping someone with a robotics question. This topic isn't covered in their current book chapters.

⚡ INSTRUCTIONS:
- Provide a SHORT answer (2-3 sentences max)
- Use your general robotics/AI knowledge
- Be concise and helpful
- Optionally mention: "This topic isn't covered in the current book chapters"

❓ QUESTION: {request.message}

💡 SHORT ANSWER:"""

            response = model.generate_content(prompt)
            
            answer_text = response.text
            if len(answer_text) > 200:
                answer_text = answer_text[:200] + "..."
            
            return ChatResponse(
                response=f"{answer_text}\n\n📝 Note: This answer is from general knowledge, not specifically from the book chapters.",
                source="general_knowledge",
                chapters=[]
            )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# -----------------------------
# 8️⃣ Health & Stats Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Physical AI Book Chatbot - Hybrid Search System", "version": "2.0"}

@app.get("/health")
async def health():
    try:
        chapters_count = len(get_all_chapters())
        api_configured = bool(os.getenv("GEMINI_API_KEY"))
        return {"status": "healthy", "book_chapters_loaded": chapters_count, "gemini_configured": api_configured}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/stats")
async def stats():
    try:
        chapters = get_all_chapters()
        return {
            "total_chapters": len(chapters),
            "difficulties": {
                "Student": len([ch for ch in chapters if ch['difficulty'] == 'Student']),
                "Professional": len([ch for ch in chapters if ch['difficulty'] == 'Professional']),
                "Researcher": len([ch for ch in chapters if ch['difficulty'] == 'Researcher'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 9️⃣ Run Locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)





# open in web page by below code
# uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs