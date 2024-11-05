from fastapi import FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import json
import os
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import jax
import jax.numpy as jnp

# Constants
NUM_CONTEXTS = 3
NUM_ARMS = 35
EXPLORATION_RATE = 0.1
SECRET_KEY = "your-secret-key"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class AnswerSubmit(BaseModel):
    username: str
    question_id: int
    answer: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Helper functions
def load_users():
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def load_questions():
    if os.path.exists('D:\\Math Mentor AI\\server\\data\\question_answer_pairs.json'):
        with open('questions.json', 'r') as f:
            return json.load(f)
    return []

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return password

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def select_question(rng_key, Q_values, context_index):
    random_value = jax.random.uniform(rng_key)
    explore = random_value < EXPLORATION_RATE
    arm_index = jnp.where(explore,
                          jax.random.randint(rng_key, (1,), 0, NUM_ARMS)[0],
                          jnp.argmax(Q_values[context_index]))
    return int(arm_index)

def custom_reward(is_correct, hints_used, timestamp_since_last, correct_ratio):
    base_reward = float(is_correct)
    hint_penalty = 0.2 * hints_used
    reward = base_reward - hint_penalty
    time_penalty = 0.5 if timestamp_since_last < 10 else 0.0
    reward -= time_penalty
    reward = reward + 0.5 * correct_ratio if is_correct else reward
    return max(reward, 0.0)

def update_q_values(Q_values, counts, context_index, arm_index, reward):
    counts[context_index][arm_index] += 1
    Q_values[context_index][arm_index] += (reward - Q_values[context_index][arm_index]) / counts[context_index][arm_index]
    return Q_values, counts

# Routes
@app.post("/signup", response_model=Token)
async def signup(user: UserCreate):
    users = load_users()
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    users[user.username] = {
        "hashed_password": hashed_password,
        "q_values": [[0.0 for _ in range(NUM_ARMS)] for _ in range(NUM_CONTEXTS)],
        "counts": [[0 for _ in range(NUM_ARMS)] for _ in range(NUM_CONTEXTS)],
        "context": {
            "timestamp_since_last": 0,
            "correct_ratio": 0,
            "total_attempts": 0,
            "correct_attempts": 0
        }
    }
    save_users(users)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/login", response_model=Token)
async def login(user: UserLogin):
    users = load_users()
    if user.username not in users or not verify_password(user.password, users[user.username]["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/question")
async def get_question(username: str):
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users[username]
    q_values = user["q_values"]
    context = user["context"]
    context_index = int(context['correct_attempts'] * (NUM_CONTEXTS - 1))
    
    rng_key = jax.random.PRNGKey(0)
    question_index = select_question(rng_key, jnp.array(q_values), context_index)
    
    questions = load_questions()
    if question_index >= len(questions):
        raise HTTPException(status_code=404, detail="Question not found")
    
    question = questions[question_index]
    return {"question": {"id": question_index, "question": question["question"]}}

@app.post("/answer")
async def submit_answer(answer: AnswerSubmit):
    users = load_users()
    if answer.username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    questions = load_questions()
    if answer.question_id >= len(questions):
        raise HTTPException(status_code=404, detail="Question not found")
    
    user = users[answer.username]
    question = questions[answer.question_id]
    
    is_correct = answer.answer.lower() == question["answer"].lower()
    context = user["context"]
    context['total_attempts'] += 1
    if is_correct:
        context['correct_attempts'] += 1
    context['correct_ratio'] = context['correct_attempts'] / context['total_attempts']
    
    reward = custom_reward(is_correct, 0, context['timestamp_since_last'], context['correct_ratio'])
    
    q_values = user["q_values"]
    counts = user["counts"]
    context_index = int(context['correct_attempts'] * (NUM_CONTEXTS - 1))
    
    q_values, counts = update_q_values(q_values, counts, context_index, answer.question_id, reward)
    
    user["q_values"] = q_values
    user["counts"] = counts
    user["context"] = context
    save_users(users)
    
    feedback = "Correct! Well done." if is_correct else "Incorrect. Try again."
    return {"correct": is_correct, "feedback": feedback}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
