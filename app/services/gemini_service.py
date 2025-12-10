import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')  # ← UPDATED!
    
    async def generate_response(self, message: str) -> str:
        """Generate AI response using Gemini"""
        try:
            prompt = f"""You are an AI assistant for the book "Physical AI and Humanoid Robotics" by Azmat Ali.
            
The book covers:
- Part 1 (Students): ROS2, Linux, URDF, Navigation, Sensors
- Part 2 (Researchers): Gazebo, Isaac Sim, Computer Vision, Reinforcement Learning
- Part 3 (Experts): Sim-to-Real, Hardware, Production, Ethics

User question: {message}

Provide a helpful, accurate, and educational response."""

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

gemini_service = GeminiService()