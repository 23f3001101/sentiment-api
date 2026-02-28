from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json, os

app = FastAPI()
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze sentiment. Respond ONLY with valid JSON: {\"sentiment\": \"positive/negative/neutral\", \"rating\": 1-5}. No extra text."
                },
                {"role": "user", "content": request.comment}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
