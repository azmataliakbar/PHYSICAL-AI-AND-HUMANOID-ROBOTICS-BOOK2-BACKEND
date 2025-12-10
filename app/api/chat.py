from fastapi import APIRouter
from app.models.chat import ChatRequest, ChatResponse
from app.services.gemini_service import gemini_service

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - connects to Gemini AI"""
    answer = await gemini_service.generate_response(request.message)
    return ChatResponse(answer=answer)