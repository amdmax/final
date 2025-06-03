from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Tuple
import main as core
from mcp_use import MCPClient, MCPAgent
import uuid
import os
import json
from dotenv import load_dotenv
import asyncio
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

API_PREFIX = "/api"
DATA_ROOT = "data"

config = {
    "mcpServers": {
        "playwright": {
            "command": "npx",
            "args": ["@playwright/mcp@latest"],
            "env": {
            "DISPLAY": ":1"
            }
        }
    }
}

# Helper to invoke MCP using async session
async def mcp_invoke(prompt: str):
    client = MCPClient.from_dict(config)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        prompt,
        max_steps=30
    )
    print(f"\nResult: {result}")
    return result

# Define generate_questions_with_tools for per-user usage (now async)
async def generate_questions_with_tools(topic: str) -> List[str]:
    prompt = f"Generate 7 questions of increasing difficulty to assess a student's knowledge of {topic}. Return as a numbered list."
    result = await mcp_invoke(prompt)
    import re
    question_lines = re.findall(r"^\s*\d+\s*[\.|\)]?\s*(.+)$", str(result), re.MULTILINE)
    questions = [q.strip() for q in question_lines if q.strip()]
    if not questions:
        questions = [q.strip() for q in str(result).split('\n') if q.strip()]
    if not questions:
        questions = [f"What is {topic}?", f"Explain a key co    ncept in {topic}."]
    return questions

# Utility to get per-user file path and ensure folder exists
def user_data_path(user_id: str, filename: str) -> str:
    user_folder = os.path.join(DATA_ROOT, str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    return os.path.join(user_folder, filename)

# Migration helper: move legacy file to user folder if needed
def migrate_legacy_file(user_id: str, filename: str):
    user_path = user_data_path(user_id, filename)
    if not os.path.exists(user_path) and os.path.exists(filename):
        os.makedirs(os.path.dirname(user_path), exist_ok=True)
        os.rename(filename, user_path)

# Per-user load/save helpers
def load_json(user_id: str, filename: str, default):
    migrate_legacy_file(user_id, filename)
    path = user_data_path(user_id, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return default

def save_json(user_id: str, filename: str, data):
    path = user_data_path(user_id, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# Update all helpers to use user_id

def load_topics(user_id: str):
    return load_json(user_id, "topics.json", {})

def save_topics(user_id: str, topics):
    save_json(user_id, "topics.json", topics)

def load_questions(user_id: str):
    return load_json(user_id, "questions.json", {})

def save_questions(user_id: str, data):
    save_json(user_id, "questions.json", data)

def load_covered(user_id: str):
    return load_json(user_id, "covered.json", {})

def save_covered(user_id: str, data):
    save_json(user_id, "covered.json", data)

def load_interactions(user_id: str):
    return load_json(user_id, "interactions.json", {})

def save_interactions(user_id: str, data):
    save_json(user_id, "interactions.json", data)

def load_progress(user_id: str):
    return load_json(user_id, "progress.json", {})

def save_progress(user_id: str, data):
    save_json(user_id, "progress.json", data)

class TopicRequest(BaseModel):
    name: str

class StudyRequest(BaseModel):
    topic: str
    answer: Optional[str] = None
    question: Optional[str] = None
    user_id: Optional[str] = "default"
    level: Optional[str] = None

# Example endpoint update (delete_topic):
# @app.delete(f"{API_PREFIX}/users/{{user_id}}/topics/{{topic}}")
# def delete_topic(user_id: str, topic: str):
#     topics = load_topics(user_id)
#     if topic in topics:
#         del topics[topic]
#         save_topics(user_id, topics)
#         return {"message": f"Deleted topic: {topic}"}
#     raise HTTPException(status_code=404, detail="Topic not found.")

@app.post(f"{API_PREFIX}/study/start")
async def start_study(req: StudyRequest):
    user_id = str(getattr(req, 'user_id', None) or "default")
    try:
        all_questions = load_questions(user_id)
        if req.topic not in all_questions:
            questions = await generate_questions_with_tools(req.topic)
            all_questions[req.topic] = questions
            save_questions(user_id, all_questions)
        else:
            questions = all_questions[req.topic]
        covered = load_covered(user_id)
        if req.topic not in covered:
            covered[req.topic] = {"covered": [], "level": 0, "session_id": str(uuid.uuid4())}
            save_covered(user_id, covered)
        return {"questions": questions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Could not start study session.", "details": str(e)})

@app.post(f"{API_PREFIX}/study/answer")
def answer_question(req: StudyRequest):
    user_id = str(getattr(req, 'user_id', None) or "default")
    try:
        covered = load_covered(user_id)
        if req.topic not in covered:
            covered[req.topic] = {"covered": [], "level": 0, "session_id": str(uuid.uuid4())}
        if req.question and req.question not in covered[req.topic]["covered"]:
            covered[req.topic]["covered"].append(req.question)
        if len(covered[req.topic]["covered"]) >= 3:
            covered[req.topic]["level"] = 1
        save_covered(user_id, covered)
        progress = load_progress(user_id)
        if req.topic not in progress:
            progress[req.topic] = {"answered": 0, "correct": 0}
        progress[req.topic]["answered"] += 1
        progress[req.topic]["correct"] += 1
        save_progress(user_id, progress)
        return {"correct": True, "llm_result": "Accepted for demo"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Could not check answer.", "details": str(e)})

@app.post(f"{API_PREFIX}/study/train")
async def train_question(req: StudyRequest):
    user_id = str(getattr(req, 'user_id', None) or "default")
    try:
        covered = load_covered(user_id)
        level = 0
        if req.topic in covered:
            level = covered[req.topic].get("level", 0)
        prompt = f"Ask a {'beginner' if level == 0 else 'intermediate'} training question about {req.topic}."
        response = await mcp_invoke(prompt)
        text = str(response)
        return {"question": text.strip()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Could not generate training question.", "details": str(e)})

@app.get(f"{API_PREFIX}/students/{{user_id}}/topics")
def get_topics(user_id: str):
    topics = load_topics(user_id)
    # Convert dict to list of {id, name} for frontend compatibility
    if isinstance(topics, dict):
        topic_list = [
            {"id": k, "name": v["name"] if isinstance(v, dict) and "name" in v else k}
            for k, v in topics.items()
        ]
    elif isinstance(topics, list):
        topic_list = [
            {"id": t["id"] if isinstance(t, dict) and "id" in t else str(t), "name": t["name"] if isinstance(t, dict) and "name" in t else str(t)}
            for t in topics
        ]
    else:
        topic_list = []
    return topic_list

@app.post(f"{API_PREFIX}/students/{{user_id}}/topics")
def add_topic(user_id: str, req: TopicRequest):
    topics = load_topics(user_id)
    # If topics is a dict, add by id; if list, append
    topic_id = req.name.strip().replace(" ", "_").lower()
    topic_obj = {"id": topic_id, "name": req.name.strip()}
    if isinstance(topics, dict):
        topics[topic_id] = topic_obj
        save_topics(user_id, topics)
    elif isinstance(topics, list):
        topics.append(topic_obj)
        save_topics(user_id, topics)
    else:
        topics = {topic_id: topic_obj}
        save_topics(user_id, topics)
    return topic_obj

@app.delete(f"{API_PREFIX}/students/{{user_id}}/topics/{{topic_id}}")
def delete_topic(user_id: str, topic_id: str):
    topics = load_topics(user_id)
    removed = False
    if isinstance(topics, dict):
        if topic_id in topics:
            del topics[topic_id]
            removed = True
            save_topics(user_id, topics)
    elif isinstance(topics, list):
        topics = [t for t in topics if not (isinstance(t, dict) and t.get("id") == topic_id)]
        removed = True
        save_topics(user_id, topics)
    if removed:
        return {"message": f"Deleted topic: {topic_id}"}
    raise HTTPException(status_code=404, detail="Topic not found.")

@app.post(f"{API_PREFIX}/students/{{user_id}}/topics/{{topic_id}}/interact")
async def interact_with_topic(user_id: str, topic_id: str, req: StudyRequest):
    topics = load_topics(user_id)
    topic_name = topic_id
    if isinstance(topics, dict) and topic_id in topics:
        topic_name = topics[topic_id]["name"] if isinstance(topics[topic_id], dict) and "name" in topics[topic_id] else topic_id
    elif isinstance(topics, list):
        for t in topics:
            if isinstance(t, dict) and t.get("id") == topic_id:
                topic_name = t.get("name", topic_id)
                break
    # Load or initialize covered and progress
    covered = load_covered(user_id)
    if topic_name not in covered:
        covered[topic_name] = {"covered": [], "level": 0, "session_id": str(uuid.uuid4())}
    progress = load_progress(user_id)
    if topic_name not in progress:
        progress[topic_name] = {"answered": 0, "correct": 0}
    # If no answer/question, start session: generate first question
    if not req.answer and not req.question:
        # Resume from last unanswered question if available
        covered_questions = covered[topic_name]["covered"]
        # If there are previously covered questions, find the next unanswered one
        if covered_questions:
            # If the last question is not fully answered, resume it
            for q in covered_questions:
                # Optionally, you could track answer status, but for now just resume the next not-yet-answered
                pass  # All covered questions are considered answered, so move to new question
        # Otherwise, generate a new question as before
        prompt = (
            f"Generate a single diagnostic question to assess a student's knowledge of {topic_name}. "
            f"Avoid repeating questions in this list: {covered_questions}. "
            f"Focus on areas that are not yet covered or where the student may have knowledge gaps. "
            f"Return only the question."
        )
        question = str(await mcp_invoke(prompt)).strip()
        # Save to covered for future resume
        covered[topic_name]["covered"].append(question)
        save_covered(user_id, covered)
        return {"questions": covered[topic_name]["covered"], "question": question}
    # Evaluate answer, give feedback, and generate next question
    feedback_prompt = (
        f"You are an adaptive learning tutor. The student answered the following question about {topic_name}:\n"
        f"Question: {req.question}\n"
        f"Student's Answer: {req.answer}\n"
        f"Evaluate the answer, give a score from 1-10, provide specific feedback, and a learning tip to help the student improve. "
        f"Respond in JSON: {{'score': <score>, 'feedback': <feedback>, 'tip': <tip>}}."
    )
    llm_response = await mcp_invoke(feedback_prompt)
    import re, json as pyjson
    try:
        match = re.search(r'\{.*\}', str(llm_response), re.DOTALL)
        if match:
            feedback_json = pyjson.loads(match.group(0).replace("'", '"'))
        else:
            # Try to extract feedback text if LLM returns a code block or extra JSON
            feedback_text = str(llm_response)
            # Remove code block markers and try to extract feedback
            feedback_text = re.sub(r'```json|```', '', feedback_text).strip()
            # Try to find a feedback string in the text
            feedback_match = re.search(r'"feedback"\s*:\s*"([^"]+)"', feedback_text)
            tip_match = re.search(r'"tip"\s*:\s*"([^"]+)"', feedback_text)
            score_match = re.search(r'"score"\s*:\s*(\d+)', feedback_text)
            feedback_json = {
                "score": int(score_match.group(1)) if score_match else 5,
                "feedback": feedback_match.group(1) if feedback_match else feedback_text,
                "tip": tip_match.group(1) if tip_match else "Keep practicing!"
            }
    except Exception:
        feedback_json = {"score": 5, "feedback": str(llm_response), "tip": "Keep practicing!"}
    # Update covered and progress
    if req.question and req.question not in covered[topic_name]["covered"]:
        covered[topic_name]["covered"].append(req.question)
    progress[topic_name]["answered"] += 1
    if feedback_json.get("score", 5) >= 7:
        progress[topic_name]["correct"] += 1
    save_covered(user_id, covered)
    save_progress(user_id, progress)
    # Generate next question, focusing on knowledge gaps
    next_q_prompt = (
        f"Given the student's previous answers and feedback, generate a new question for {topic_name} "
        f"that explores areas not yet covered or where the student showed weakness.\n"
        f"Already covered: {covered[topic_name]['covered']}\n"
        f"Last answer: {req.answer}\n"
        f"Last feedback: {feedback_json.get('feedback', '')}\n"
        f"Return only the question."
    )
    next_question = str(await mcp_invoke(next_q_prompt)).strip()
    # If the LLM repeats a previous question, try to generate a new one
    if next_question in covered[topic_name]["covered"]:
        alt_prompt = (
            f"Generate a new, unique question for {topic_name} that is not in this list: {covered[topic_name]['covered']}. "
            f"Return only the question."
        )
        next_question = str(await mcp_invoke(alt_prompt)).strip()
    return {
        "action": "ask_question",
        "question": next_question,
        "last_feedback": feedback_json.get("feedback", "Good try!"),
        "learning_tip": feedback_json.get("tip", "Keep practicing!"),
        "score": feedback_json.get("score", 5),
        "questions": covered[topic_name]["covered"] + [next_question],
        "message": "Here's your feedback and a new question!"
    }
