from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Tuple
import main as core
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import requests
import uuid
import os
import json

app = FastAPI()

API_PREFIX = "/api"

# Data files
TOPICS_FILE = "topics.json"
PROGRESS_FILE = "progress.json"
QUESTIONS_FILE = "questions.json"
COVERED_FILE = "covered.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")
if not PERPLEXITY_API_KEY:
    # If Perplexity is required elsewhere, you can raise or warn here
    import warnings
    warnings.warn("PERPLEXITY_API_KEY environment variable not set. Some features may not work.")

# Initialize model for OpenPerplexity/LLM use
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    model_kwargs={"truncation": "auto"},
    api_key=OPENAI_API_KEY,
)

class TopicRequest(BaseModel):
    name: str

class StudyRequest(BaseModel):
    topic: str
    answer: Optional[str] = None
    question: Optional[str] = None
    level: Optional[str] = None

# Helper to persist/load questions per topic

def load_questions():
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_questions(data):
    with open(QUESTIONS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_covered():
    if os.path.exists(COVERED_FILE):
        with open(COVERED_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_covered(data):
    with open(COVERED_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# New: Load/save topics as objects with id and name
# Topics are now per-student: store topics as {student_id: [topic, ...]}
def load_topics():
    if os.path.exists(TOPICS_FILE):
        with open(TOPICS_FILE, 'r') as f:
            data = json.load(f)
            # Migrate from old format if needed
            if isinstance(data, list):
                # Convert to {student_id: [{id, name}]}
                data = {"default": [{"id": str(uuid.uuid4()), "name": name} for name in data]}
                save_topics(data)
            return data
    return {}

def save_topics(topics):
    with open(TOPICS_FILE, 'w') as f:
        json.dump(topics, f, indent=2)

# New: Per-student/topic interaction memory
def load_interactions():
    path = 'interactions.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_interactions(data):
    with open('interactions.json', 'w') as f:
        json.dump(data, f, indent=2)

# OpenPerplexity integration

# Update generate_questions_openperplexity to accept topic_name

def generate_questions_openperplexity(topic: str) -> List[str]:
    prompt = PromptTemplate.from_template("Generate 7 questions of increasing difficulty to assess a student's knowledge of {topic}. Return as a numbered list.\n")
    chain = prompt | llm
    result = chain.invoke({"topic": topic})
    # result.content is a list of dicts with 'text' key, or a string
    if hasattr(result, 'content') and isinstance(result.content, list):
        # Find the first text block
        for part in result.content:
            if isinstance(part, dict) and 'text' in part:
                text = part['text']
                break
        else:
            text = str(result)
    elif hasattr(result, 'content') and isinstance(result.content, str):
        text = result.content
    else:
        text = str(result)
    import re
    question_lines = re.findall(r"^\s*\d+\s*[\.|\)]?\s*(.+)$", text, re.MULTILINE)
    questions = [q.strip() for q in question_lines if q.strip()]
    if not questions:
        # fallback: try splitting by newlines if LLM didn't number
        questions = [q.strip() for q in text.split('\n') if q.strip()]
    if not questions:
        questions = [f"What is {topic}?", f"Explain a key concept in {topic}."]
    return questions
    

@app.delete(f"{API_PREFIX}/topics/{{topic}}")
def delete_topic(topic: str):
    topics = core.load_json(core.TOPICS_FILE, [])
    if topic in topics:
        topics.remove(topic)
        core.save_json(core.TOPICS_FILE, topics)
        return {"message": f"Deleted topic: {topic}"}
    raise HTTPException(status_code=404, detail="Topic not found.")

@app.post(f"{API_PREFIX}/study/start")
def start_study(req: StudyRequest):
    try:
        all_questions = load_questions()
        if req.topic not in all_questions:
            # Generate questions if not present
            questions = generate_questions_openperplexity(req.topic)
            all_questions[req.topic] = questions
            save_questions(all_questions)
        else:
            questions = all_questions[req.topic]
        # Mark as started in covered.json
        covered = load_covered()
        if req.topic not in covered:
            covered[req.topic] = {"covered": [], "level": 0, "session_id": str(uuid.uuid4())}
            save_covered(covered)
        return {"questions": questions}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not start study session. This may be due to a missing or invalid OpenPerplexity API key or another configuration issue.",
                "details": str(e)
            }
        )

@app.post(f"{API_PREFIX}/study/answer")
def answer_question(req: StudyRequest):
    try:
        # For demo: accept any answer as correct, but mark as covered
        covered = load_covered()
        if req.topic not in covered:
            covered[req.topic] = {"covered": [], "level": 0, "session_id": str(uuid.uuid4())}
        if req.question and req.question not in covered[req.topic]["covered"]:
            covered[req.topic]["covered"].append(req.question)
        # Optionally, increase level if enough questions are covered
        if len(covered[req.topic]["covered"]) >= 3:
            covered[req.topic]["level"] = 1
        save_covered(covered)
        # Track progress
        core.track_progress(req.topic, True)
        return {"correct": True, "llm_result": "Accepted for demo"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not check answer.",
                "details": str(e)
            }
        )

@app.post(f"{API_PREFIX}/study/train")
def train_question(req: StudyRequest):
    try:
        # Generate a new context or question for the student, increasing complexity
        covered = load_covered()
        level = 0
        if req.topic in covered:
            level = covered[req.topic].get("level", 0)
        prompt = f"Ask a {'beginner' if level == 0 else 'intermediate'} training question about {req.topic}."
        response = llm.invoke([
            {
                "role": "user",
                "content": prompt
            }
        ])
        text = response[0]["content"] if isinstance(response, list) and response and "content" in response[0] else str(response)
        return {"question": text.strip()}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not generate training question.",
                "details": str(e)
            }
        )


class Agent:
    """
    Agent manages student state, decides next actions, and orchestrates adaptive learning.
    State is persisted in covered.json and progress.json.
    Now supports LLM-based answer scoring (1-10) and level-up at 20/40/70 points.
    """
    SYSTEM_PROMPT = (
        """
        ROLE: You are a helpful training assistant.\n"
        "ACTIONS:\n"
        "- Research the topic and provide clear, engaging explanations, interesting facts, and stories.\n"
        "- Evaluate the student's level of understanding on the topic.\n"
        "- Keep pushing the student forward by sharing new information, asking follow-up questions, and encouraging curiosity.\n"
        "- After each interaction, score and grade the student, and display their current grade.\n"
        "- Use the student's previous answers and your own previous messages to adapt your teaching and evaluation.\n"
        "- If the student is making progress, increase the difficulty or depth of the material.\n"
        "- If the student is struggling, offer simpler explanations, analogies, or additional resources.\n"
        "- Never stop the session; always continue the conversation, unless the student explicitly asks to stop.\n"
        """
    )

    def __init__(self, user_id: str = "default", topic_name_map=None):
        self.user_id = str(user_id) if user_id is not None else "default"
        self.covered = load_covered()
        self.progress = core.load_json(core.PROGRESS_FILE, {})
        self.questions = load_questions()
        self.topic_name_map = topic_name_map or (lambda tid: tid)
        self.scores = self.progress.get("scores", {})  # {topic: int}

    def get_state(self, topic: str):
        topic = str(topic) if topic is not None else ""
        if topic not in self.covered:
            self.covered[topic] = {"covered": [], "level": 0, "session_id": str(uuid.uuid4())}
        return self.covered[topic]

    def save_state(self):
        save_covered(self.covered)
        core.save_json(core.PROGRESS_FILE, self.progress)

    def get_score(self, topic: str):
        if "scores" not in self.progress:
            self.progress["scores"] = {}
        return self.progress["scores"].get(topic, 0)

    def add_score(self, topic: str, score: int):
        if "scores" not in self.progress:
            self.progress["scores"] = {}
        self.progress["scores"][topic] = self.progress["scores"].get(topic, 0) + score

    def generate_learning_tip(self, topic_name: str, last_feedback: str = "") -> str:
        import re
        prompt = f"""
        You are an expert tutor. Based on the student's recent answer and feedback, suggest a short, actionable learning tip or resource for improving their knowledge of {topic_name}. {last_feedback}
        """
        response = llm.invoke([
            {"role": "user", "content": prompt}
        ])
        # Robustly extract only the tip text from various possible LLM response formats
        text = None
        # If response is a list of dicts with 'text', extract the first 'text'
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict):
                if 'text' in first and isinstance(first['text'], str):
                    text = first['text']
                elif 'content' in first and isinstance(first['content'], str):
                    text = first['content']
        elif isinstance(response, dict):
            if 'text' in response and isinstance(response['text'], str):
                text = response['text']
            elif 'content' in response and isinstance(response['content'], str):
                text = response['content']
        if not text:
            # Try to extract 'text' from a string like "content=[{{'type': 'text', 'text': '...'}}] ..."
            s = str(response)
            # Try to find the first 'text': '...' value
            match = re.search(r"'text': '([^']+)'", s)
            if match:
                text = match.group(1)
            else:
                # Try to find the first quoted sentence after a colon
                match2 = re.search(r":\s*['\"]([^'\"]+)['\"]", s)
                if match2:
                    text = match2.group(1)
                else:
                    text = s
        # If the text contains a long prefix (e.g., content=[...]), try to extract the first sentence or the part starting with a capital letter and a verb
        # Try to find the first sentence that looks like a tip (e.g., starts with 'To ', 'Try ', 'Consider ', etc.)
        tip_match = re.search(r'(To [^.]+\.|Try [^.]+\.|Consider [^.]+\.|You should [^.]+\.|[A-Z][^.]+\.)', text)
        if tip_match:
            return tip_match.group(1).strip()
        # Otherwise, just return the text, stripped
        return text.strip()

    def decide_next(self, topic: str, last_answer: Optional[str] = None, last_question: Optional[str] = None, last_score: Optional[int] = None, last_feedback: Optional[str] = None):
        topic = str(topic) if topic is not None else ""
        state = self.get_state(topic)
        level = state.get("level", 0)
        covered_qs = state.get("covered", [])
        topic_name = self.topic_name_map(topic)
        # If no questions yet, generate them
        if topic not in self.questions:
            self.questions[topic] = generate_questions_openperplexity(topic_name)
        # Find next uncovered question
        for q in self.questions[topic]:
            if q not in covered_qs:
                # If the question contains the topic id, replace with topic name
                if topic in q:
                    q = q.replace(topic, topic_name)
                return {"action": "ask_question", "question": q, "level": level, "score": self.get_score(topic), "last_score": last_score, "last_feedback": last_feedback}
        # If all covered, escalate level or give feedback
        thresholds = [20, 40, 70]
        score = self.get_score(topic)
        if level < len(thresholds) and score >= thresholds[level]:
            state["level"] = level + 1
            self.save_state()
            return {"action": "level_up", "message": f"Level up! Now at level {level+1} for {topic_name}.", "score": score}
        return {"action": "complete", "message": f"All questions for {topic_name} completed! Final score: {score}/100", "score": score}

    def score_answer(self, question: str, answer: str, topic_name: str) -> Tuple[int, str]:
        prompt = (
            f"You are an expert tutor. Evaluate the following student's answer to the question about {topic_name}.\n"
            f"Question: {question}\n"
            f"Student Answer: {answer}\n"
            f"Give a score from 1 (very poor) to 10 (excellent) and a short constructive feedback. Respond in JSON: {{\"score\": <int>, \"feedback\": \"<string>\"}}"
        )
        try:
            response = llm.invoke([
                {"role": "user", "content": prompt}
            ])
            import re, json as pyjson
            # If response is a list of dicts with 'text', extract the text
            if isinstance(response, list) and response and isinstance(response[0], dict) and 'text' in response[0]:
                text = response[0]['text']
            elif isinstance(response, list) and response and 'content' in response[0]:
                text = response[0]['content']
            else:
                text = str(response)
            # Try to extract JSON from the text
            json_match = re.search(r'\{[\s\S]*?\}', text)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = pyjson.loads(json_str)
                    score = int(data.get("score", 1))
                    feedback = data.get("feedback", "")
                    if not isinstance(score, int) or score < 1 or score > 10:
                        score = 1
                    if not feedback or not isinstance(feedback, str):
                        feedback = "No feedback provided."
                    return score, feedback
                except Exception:
                    pass
            # Fallback: if no JSON, just return 5 and the raw text
            return 5, text.strip()
        except Exception as e:
            return 5, f"Could not evaluate answer: {e}"

    def process(self, input: dict):
        """
        Main agentic interaction method. Handles user input and returns next action/response.
        input: {topic, answer, question, ...}
        """
        topic = str(input.get("topic")) if input.get("topic") is not None else ""
        answer = input.get("answer")
        question = input.get("question")
        # Defensive: ensure topic is a string
        if not isinstance(topic, str):
            topic = str(topic)
        last_score = None
        last_feedback = None
        # Load and persist conversation history for this student/topic
        interactions = load_interactions()
        if self.user_id not in interactions:
            interactions[self.user_id] = {}
        if topic not in interactions[self.user_id]:
            interactions[self.user_id][topic] = []
        # Score the answer if present
        if answer and question:
            topic_name = self.topic_name_map(topic)
            score, feedback = self.score_answer(question, answer, topic_name)
            self.add_score(topic, score)
            last_score = score
            # --- Clean up feedback: if feedback contains embedded JSON, extract just the feedback string ---
            import re, json as pyjson
            feedback_str = feedback
            # Try to extract JSON from feedback if it looks like a dict or contains 'text':
            json_match = re.search(r'\{[\s\S]*?\}', feedback)
            if json_match:
                try:
                    data = pyjson.loads(json_match.group(0))
                    if isinstance(data, dict) and 'feedback' in data:
                        feedback_str = data['feedback']
                except Exception:
                    pass
            # Try to extract 'text' field if present (for OpenAI/LLM response wrappers)
            text_match = re.search(r"'text': '([^']+)'", feedback)
            if text_match:
                inner_json_match = re.search(r'\{[\s\S]*?\}', text_match.group(1))
                if inner_json_match:
                    try:
                        data = pyjson.loads(inner_json_match.group(0))
                        if isinstance(data, dict) and 'feedback' in data:
                            feedback_str = data['feedback']
                    except Exception:
                        pass
            # Fallback: if feedback_str is still the full string, try to extract the first quoted sentence
            if feedback_str == feedback:
                quote_match = re.search(r'"feedback":\s*"([^"]+)"', feedback)
                if quote_match:
                    feedback_str = quote_match.group(1)
            last_feedback = feedback_str
            # Add the latest user answer to the history, including score and feedback (just the feedback string)
            interaction_entry = {"role": "user", "content": answer, "score": score, "feedback": feedback_str}
            interactions[self.user_id][topic].append(interaction_entry)
        elif answer:
            # If no question, just log the answer
            interactions[self.user_id][topic].append({"role": "user", "content": answer})
        # If all questions are covered, switch to exploration mode
        state = self.get_state(topic)
        covered_qs = state.get("covered", [])
        if topic in self.questions and len(covered_qs) >= len(self.questions[topic]):
            # Exploration mode: use system prompt and conversation history
            history = interactions[self.user_id][topic][-10:]  # last 10 turns
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Let's continue learning about {self.topic_name_map(topic)}. My last answer: {answer or ''}"}
            ]
            for h in history:
                messages.append(h)
            # Ask the LLM to continue the session
            response = llm.invoke(messages)
            text = response[0]["content"] if isinstance(response, list) and response and "content" in response[0] else str(response)
            # Score the answer if possible
            score, feedback = self.score_answer(question or "", answer or text, self.topic_name_map(topic))
            self.add_score(topic, score)
            self.save_state()
            # Save the assistant's message to history, including score/feedback
            interactions[self.user_id][topic].append({
                "role": "assistant",
                "content": text,
                "score": score,
                "feedback": feedback
            })
            save_interactions(interactions)
            # Add learning tip
            learning_tip = self.generate_learning_tip(self.topic_name_map(topic), feedback)
            return {
                "action": "explore",
                "message": text,
                "score": self.get_score(topic),
                "last_score": score,
                "last_feedback": feedback,
                "learning_tip": learning_tip
            }
        # Mark question as covered if answered (already handled above if answer and question)
        if question and answer and topic in self.covered and question not in self.covered.get(topic, {}).get("covered", []):
            self.covered[topic]["covered"].append(question)
            # Track progress in the same way as main.py
            if topic not in self.progress or not isinstance(self.progress[topic], dict):
                self.progress[topic] = {"answered": 0, "correct": 0}
            self.progress[topic]["answered"] += 1
            self.progress[topic]["correct"] += 1
        self.save_state()
        save_interactions(interactions)
        # Decide next action and always include score, feedback, and learning_tip in response
        next_action = self.decide_next(topic, last_answer=answer, last_question=question, last_score=last_score, last_feedback=last_feedback)
        # Add score, feedback, and learning_tip to the response
        next_action["score"] = self.get_score(topic)
        next_action["last_score"] = last_score
        next_action["last_feedback"] = last_feedback
        next_action["learning_tip"] = self.generate_learning_tip(self.topic_name_map(topic), last_feedback or "")
        return next_action


class AgentRequest(BaseModel):
    topic: str
    answer: Optional[str] = None
    question: Optional[str] = None
    user_id: Optional[str] = "default"

@app.post(f"{API_PREFIX}/agent/interact")
def agent_interact(req: AgentRequest):
    try:
        # Defensive: ensure topic and user_id are always strings
        topic = str(req.topic) if req.topic is not None else ""
        user_id = str(req.user_id) if req.user_id is not None else "default"
        agent = Agent(user_id=user_id)
        result = agent.process({**req.dict(), "topic": topic, "user_id": user_id})
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Agentic interaction failed.",
                "details": str(e)
            }
        )

@app.post(f"{API_PREFIX}/students/{{student_id}}/topics")
def add_topic(student_id: str, req: TopicRequest):
    topics = load_topics()
    if student_id not in topics:
        topics[student_id] = []
    # Check for duplicate by name
    for t in topics[student_id]:
        if t["name"].lower() == req.name.lower():
            return JSONResponse(status_code=400, content={"error": "Topic already exists."})
    topic_id = str(uuid.uuid4())
    new_topic = {"id": topic_id, "name": req.name}
    topics[student_id].append(new_topic)
    save_topics(topics)
    # Initialize questions for this topic for this student
    all_questions = load_questions()
    if student_id not in all_questions:
        all_questions[student_id] = {}
    all_questions[student_id][topic_id] = generate_questions_openperplexity(req.name)
    save_questions(all_questions)
    # Initialize progress and covered for this student/topic
    progress = core.load_json(PROGRESS_FILE, {})
    if student_id not in progress:
        progress[student_id] = {}
    progress[student_id][topic_id] = {"answered": 0, "correct": 0}
    core.save_json(PROGRESS_FILE, progress)
    covered = load_covered()
    if student_id not in covered:
        covered[student_id] = {}
    covered[student_id][topic_id] = {"covered": [], "level": 0, "session_id": str(uuid.uuid4())}
    save_covered(covered)
    return {"id": topic_id, "name": req.name}

@app.get(f"{API_PREFIX}/students/{{student_id}}/topics")
def get_topics(student_id: str):
    topics = load_topics()
    return topics.get(student_id, [])

@app.get(f"{API_PREFIX}/students/{{student_id}}/topics/{{topic_id}}/questions")
def get_questions(student_id: str, topic_id: str):
    all_questions = load_questions()
    return all_questions.get(student_id, {}).get(topic_id, [])

@app.get(f"{API_PREFIX}/students/{{student_id}}/topics/{{topic_id}}/progress")
def get_progress(student_id: str, topic_id: str):
    progress = core.load_json(PROGRESS_FILE, {})
    return progress.get(student_id, {}).get(topic_id, {"answered": 0, "correct": 0})

@app.get(f"{API_PREFIX}/students/{{student_id}}/topics/{{topic_id}}/interactions")
def get_interactions(student_id: str, topic_id: str):
    interactions = load_interactions()
    return interactions.get(student_id, {}).get(topic_id, [])

# Helper: Map topic_id to topic name for a student
def get_topic_name(student_id: str, topic_id: str) -> str:
    topics = load_topics()
    for t in topics.get(student_id, []):
        if t["id"] == topic_id:
            return t["name"]
    return topic_id  # fallback to id if not found

class AdaptiveLearningAgent:
    """
    An agent that interacts with the user, assesses knowledge, and provides adaptive learning and feedback.
    It uses LLM to generate questions, evaluate answers, and suggest next steps or learning resources.
    """
    SYSTEM_PROMPT = (
        """
        ROLE: You are a helpful training assistant.\n"
        "ACTIONS:\n"
        "- Research the topic and provide clear, engaging explanations, interesting facts, and stories.\n"
        "- Evaluate the student's level of understanding on the topic.\n"
        "- Keep pushing the student forward by sharing new information, asking follow-up questions, and encouraging curiosity.\n"
        "- After each interaction, score and grade the student, and display their current grade.\n"
        "- Use the student's previous answers and your own previous messages to adapt your teaching and evaluation.\n"
        "- If the student is making progress, increase the difficulty or depth of the material.\n"
        "- If the student is struggling, offer simpler explanations, analogies, or additional resources.\n"
        "- Never stop the session; always continue the conversation, unless the student explicitly asks to stop.\n"
        """
    )

    def __init__(self, user_id: str = "default", topic_name_map=None):
        self.user_id = str(user_id) if user_id is not None else "default"
        self.covered = load_covered()
        self.progress = core.load_json(core.PROGRESS_FILE, {})
        self.questions = load_questions()
        self.topic_name_map = topic_name_map or (lambda tid: tid)
        self.scores = self.progress.get("scores", {})

    def get_state(self, topic: str):
        topic = str(topic) if topic is not None else ""
        if topic not in self.covered:
            self.covered[topic] = {"covered": [], "level": 0, "session_id": str(uuid.uuid4())}
        return self.covered[topic]

    def save_state(self):
        save_covered(self.covered)
        core.save_json(core.PROGRESS_FILE, self.progress)

    def get_score(self, topic: str):
        if "scores" not in self.progress:
            self.progress["scores"] = {}
        return self.progress["scores"].get(topic, 0)

    def add_score(self, topic: str, score: int):
        if "scores" not in self.progress:
            self.progress["scores"] = {}
        self.progress["scores"][topic] = self.progress["scores"].get(topic, 0) + score

    def generate_learning_tip(self, topic_name: str, last_feedback: str = "") -> str:
        import re
        prompt = f"""
        You are an expert tutor. Based on the student's recent answer and feedback, suggest a short, actionable learning tip or resource for improving their knowledge of {topic_name}. {last_feedback}
        """
        response = llm.invoke([
            {"role": "user", "content": prompt}
        ])
        # Robustly extract only the tip text from various possible LLM response formats
        text = None
        # If response is a list of dicts with 'text', extract the first 'text'
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict):
                if 'text' in first and isinstance(first['text'], str):
                    text = first['text']
                elif 'content' in first and isinstance(first['content'], str):
                    text = first['content']
        elif isinstance(response, dict):
            if 'text' in response and isinstance(response['text'], str):
                text = response['text']
            elif 'content' in response and isinstance(response['content'], str):
                text = response['content']
        if not text:
            # Try to extract 'text' from a string like "content=[{{'type': 'text', 'text': '...'}}] ..."
            s = str(response)
            # Try to find the first 'text': '...' value
            match = re.search(r"'text': '([^']+)'", s)
            if match:
                text = match.group(1)
            else:
                # Try to find the first quoted sentence after a colon
                match2 = re.search(r":\s*['\"]([^'\"]+)['\"]", s)
                if match2:
                    text = match2.group(1)
                else:
                    text = s
        # If the text contains a long prefix (e.g., content=[...]), try to extract the first sentence or the part starting with a capital letter and a verb
        # Try to find the first sentence that looks like a tip (e.g., starts with 'To ', 'Try ', 'Consider ', etc.)
        tip_match = re.search(r'(To [^.]+\.|Try [^.]+\.|Consider [^.]+\.|You should [^.]+\.|[A-Z][^.]+\.)', text)
        if tip_match:
            return tip_match.group(1).strip()
        # Otherwise, just return the text, stripped
        return text.strip()

    def decide_next(self, topic: str, last_answer: Optional[str] = None, last_question: Optional[str] = None, last_score: Optional[int] = None, last_feedback: Optional[str] = None):
        topic = str(topic) if topic is not None else ""
        state = self.get_state(topic)
        level = state.get("level", 0)
        covered_qs = state.get("covered", [])
        topic_name = self.topic_name_map(topic)
        # If no questions yet, generate them
        if topic not in self.questions:
            self.questions[topic] = generate_questions_openperplexity(topic_name)
        # Find next uncovered question
        for q in self.questions[topic]:
            if q not in covered_qs:
                if topic in q:
                    q = q.replace(topic, topic_name)
                # Provide a learning tip if there was feedback
                learning_tip = self.generate_learning_tip(topic_name, last_feedback or "") if last_feedback else None
                return {
                    "action": "ask_question",
                    "question": q,
                    "level": level,
                    "score": self.get_score(topic),
                    "last_score": last_score,
                    "last_feedback": last_feedback,
                    "learning_tip": learning_tip
                }
        # If all covered, escalate level or give feedback
        thresholds = [20, 40, 70]
        score = self.get_score(topic)
        if level < len(thresholds) and score >= thresholds[level]:
            state["level"] = level + 1
            self.save_state()
            return {"action": "level_up", "message": f"Level up! Now at level {level+1} for {topic_name}.", "score": score}
        # Provide a final learning tip
        learning_tip = self.generate_learning_tip(topic_name, last_feedback or "") if last_feedback else None
        return {
            "action": "complete",
            "message": f"All questions for {topic_name} completed! Final score: {score}/100",
            "score": score,
            "learning_tip": learning_tip
        }

    def score_answer(self, question: str, answer: str, topic_name: str) -> Tuple[int, str]:
        prompt = (
            f"You are an expert tutor. Evaluate the following student's answer to the question about {topic_name}.\n"
            f"Question: {question}\n"
            f"Student Answer: {answer}\n"
            f"Give a score from 1 (very poor) to 10 (excellent) and a short constructive feedback. Respond in JSON: {{\"score\": <int>, \"feedback\": \"<string>\"}}"
        )
        try:
            response = llm.invoke([
                {"role": "user", "content": prompt}
            ])
            import re, json as pyjson
            # If response is a list of dicts with 'text', extract the text
            if isinstance(response, list) and response and isinstance(response[0], dict) and 'text' in response[0]:
                text = response[0]['text']
            elif isinstance(response, list) and response and 'content' in response[0]:
                text = response[0]['content']
            else:
                text = str(response)
            # Try to extract JSON from the text
            json_match = re.search(r'\{[\s\S]*?\}', text)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = pyjson.loads(json_str)
                    score = int(data.get("score", 1))
                    feedback = data.get("feedback", "")
                    if not isinstance(score, int) or score < 1 or score > 10:
                        score = 1
                    if not feedback or not isinstance(feedback, str):
                        feedback = "No feedback provided."
                    return score, feedback
                except Exception:
                    pass
            # Fallback: if no JSON, just return 5 and the raw text
            return 5, text.strip()
        except Exception as e:
            return 5, f"Could not evaluate answer: {e}"

    def process(self, input: dict):
        """
        Main agentic interaction method. Handles user input and returns next action/response.
        input: {topic, answer, question, ...}
        """
        topic = str(input.get("topic")) if input.get("topic") is not None else ""
        answer = input.get("answer")
        question = input.get("question")
        # Defensive: ensure topic is a string
        if not isinstance(topic, str):
            topic = str(topic)
        last_score = None
        last_feedback = None
        # Load and persist conversation history for this student/topic
        interactions = load_interactions()
        if self.user_id not in interactions:
            interactions[self.user_id] = {}
        if topic not in interactions[self.user_id]:
            interactions[self.user_id][topic] = []
        # Score the answer if present
        if answer and question:
            topic_name = self.topic_name_map(topic)
            score, feedback = self.score_answer(question, answer, topic_name)
            self.add_score(topic, score)
            last_score = score
            # --- Clean up feedback: if feedback contains embedded JSON, extract just the feedback string ---
            import re, json as pyjson
            feedback_str = feedback
            # Try to extract JSON from feedback if it looks like a dict or contains 'text':
            json_match = re.search(r'\{[\s\S]*?\}', feedback)
            if json_match:
                try:
                    data = pyjson.loads(json_match.group(0))
                    if isinstance(data, dict) and 'feedback' in data:
                        feedback_str = data['feedback']
                except Exception:
                    pass
            # Try to extract 'text' field if present (for OpenAI/LLM response wrappers)
            text_match = re.search(r"'text': '([^']+)'", feedback)
            if text_match:
                inner_json_match = re.search(r'\{[\s\S]*?\}', text_match.group(1))
                if inner_json_match:
                    try:
                        data = pyjson.loads(inner_json_match.group(0))
                        if isinstance(data, dict) and 'feedback' in data:
                            feedback_str = data['feedback']
                    except Exception:
                        pass
            # Fallback: if feedback_str is still the full string, try to extract the first quoted sentence
            if feedback_str == feedback:
                quote_match = re.search(r'"feedback":\s*"([^"]+)"', feedback)
                if quote_match:
                    feedback_str = quote_match.group(1)
            last_feedback = feedback_str
            # Add the latest user answer to the history, including score and feedback (just the feedback string)
            interaction_entry = {"role": "user", "content": answer, "score": score, "feedback": feedback_str}
            interactions[self.user_id][topic].append(interaction_entry)
        elif answer:
            # If no question, just log the answer
            interactions[self.user_id][topic].append({"role": "user", "content": answer})
        # If all questions are covered, switch to exploration mode
        state = self.get_state(topic)
        covered_qs = state.get("covered", [])
        if topic in self.questions and len(covered_qs) >= len(self.questions[topic]):
            # Exploration mode: use system prompt and conversation history
            history = interactions[self.user_id][topic][-10:]  # last 10 turns
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Let's continue learning about {self.topic_name_map(topic)}. My last answer: {answer or ''}"}
            ]
            for h in history:
                messages.append(h)
            # Ask the LLM to continue the session
            response = llm.invoke(messages)
            text = response[0]["content"] if isinstance(response, list) and response and "content" in response[0] else str(response)
            # Score the answer if possible
            score, feedback = self.score_answer(question or "", answer or text, self.topic_name_map(topic))
            self.add_score(topic, score)
            self.save_state()
            # Save the assistant's message to history, including score/feedback
            interactions[self.user_id][topic].append({
                "role": "assistant",
                "content": text,
                "score": score,
                "feedback": feedback
            })
            save_interactions(interactions)
            # Add learning tip
            learning_tip = self.generate_learning_tip(self.topic_name_map(topic), feedback)
            return {
                "action": "explore",
                "message": text,
                "score": self.get_score(topic),
                "last_score": score,
                "last_feedback": feedback,
                "learning_tip": learning_tip
            }
        # Mark question as covered if answered (already handled above if answer and question)
        if question and answer and topic in self.covered and question not in self.covered.get(topic, {}).get("covered", []):
            self.covered[topic]["covered"].append(question)
            # Track progress in the same way as main.py
            if topic not in self.progress or not isinstance(self.progress[topic], dict):
                self.progress[topic] = {"answered": 0, "correct": 0}
            self.progress[topic]["answered"] += 1
            self.progress[topic]["correct"] += 1
        self.save_state()
        save_interactions(interactions)
        # Decide next action and always include score, feedback, and learning_tip in response
        next_action = self.decide_next(topic, last_answer=answer, last_question=question, last_score=last_score, last_feedback=last_feedback)
        # Add score, feedback, and learning_tip to the response
        next_action["score"] = self.get_score(topic)
        next_action["last_score"] = last_score
        next_action["last_feedback"] = last_feedback
        next_action["learning_tip"] = self.generate_learning_tip(self.topic_name_map(topic), last_feedback or "")
        return next_action
# --- AGENTIC ENDPOINT ---

@app.post(f"{API_PREFIX}/students/{{student_id}}/topics/{{topic_id}}/interact")
def agentic_interact(student_id: str, topic_id: str, req: AgentRequest):
    try:
        interactions = load_interactions()
        if student_id not in interactions:
            interactions[student_id] = {}
        if topic_id not in interactions[student_id]:
            interactions[student_id][topic_id] = []
        interactions[student_id][topic_id].append(req.dict())
        save_interactions(interactions)
        topic_name = get_topic_name(student_id, topic_id)
        agent = AdaptiveLearningAgent(user_id=student_id, topic_name_map=lambda tid: get_topic_name(student_id, tid))
        agent.covered = load_covered().get(student_id, {})
        agent.progress = core.load_json(PROGRESS_FILE, {}).get(student_id, {})
        agent.questions = load_questions().get(student_id, {})
        result = agent.process({**req.dict(), "topic": topic_id, "user_id": student_id})
        covered = load_covered()
        if student_id not in covered:
            covered[student_id] = {}
        covered[student_id][topic_id] = agent.covered.get(topic_id, {"covered": [], "level": 0, "session_id": str(uuid.uuid4())})
        save_covered(covered)
        progress = core.load_json(PROGRESS_FILE, {})
        if student_id not in progress:
            progress[student_id] = {}
        progress[student_id][topic_id] = agent.progress.get(topic_id, {"answered": 0, "correct": 0})
        core.save_json(PROGRESS_FILE, progress)
        all_questions = load_questions()
        if student_id not in all_questions:
            all_questions[student_id] = {}
        all_questions[student_id][topic_id] = agent.questions.get(topic_id, [])
        save_questions(all_questions)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Agentic interaction failed.",
                "details": str(e)
            }
        )
