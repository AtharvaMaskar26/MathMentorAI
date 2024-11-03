from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import random
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from deepgram import DeepgramClient, SpeakOptions
import os
import uuid
from dotenv import load_dotenv

from utils import *
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:5173",  # Your frontend's URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Deepgram client
deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

def load_questions():
    with open('D:\\Math Mentor AI\\server\\data\\qestion_answer_pairs.json', 'r') as f:
        return json.load(f)

@app.get("/question")
async def get_question():
    questions = load_questions()
    return random.choice(questions)

class AnswerRequest(BaseModel):  # Define the model for request
    user_answer: str
    question_id: int

@app.post("/answer")
async def check_answer(answer_request: AnswerRequest):  # Use the model here
    questions = load_questions()
    question = next((q for q in questions if q["id"] == answer_request.question_id), None)

    if question:
        original_question = question.get('question')
        ground_truth = question.get('ground_truth')

        # Check if the answer is correct
        is_answer_correct = check_student_answer(answer_request.user_answer, original_question, ground_truth)

        return {"message": is_answer_correct}
    return {"message": "Question not found"}

class SpeakRequest(BaseModel):
    text: str

@app.post("/speak")
async def speak(speak_request: SpeakRequest):
    filename = f"output-{uuid.uuid4()}.wav"  # Unique filename
    try:
        # Configure the options for Deepgram speech synthesis
        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )

        # Generate the audio file using Deepgram
        response = deepgram.speak.v("1").save(filename, {"text": speak_request.text}, options)

        # Wait for the audio to finish processing
        response.wait()  # Optionally wait for the response to complete if needed

        return JSONResponse(content={"message": "Audio generated", "filename": filename})

    except Exception as e:
        print(f"Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

