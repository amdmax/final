import os
import json
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

TOPICS_FILE = "topics.json"
PROGRESS_FILE = "progress.json"

# --- Data Management ---
def load_json(filename, default):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return default

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def list_topics():
    topics = load_json(TOPICS_FILE, [])
    if not topics:
        print("No topics found.")
    for idx, topic in enumerate(topics, 1):
        print(f"{idx}. {topic}")
    return topics

def add_topic():
    topic = input("Enter new topic name: ").strip()
    topics = load_json(TOPICS_FILE, [])
    if topic and topic not in topics:
        topics.append(topic)
        save_json(TOPICS_FILE, topics)
        print(f"Added topic: {topic}")
    else:
        print("Topic already exists or invalid.")

def delete_topic():
    topics = load_json(TOPICS_FILE, [])
    list_topics()
    idx = input("Enter topic number to delete: ")
    try:
        idx = int(idx) - 1
        if 0 <= idx < len(topics):
            removed = topics.pop(idx)
            save_json(TOPICS_FILE, topics)
            print(f"Deleted topic: {removed}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")

def track_progress(topic, correct):
    progress = load_json(PROGRESS_FILE, {})
    if topic not in progress:
        progress[topic] = {"answered": 0, "correct": 0}
    progress[topic]["answered"] += 1
    if correct:
        progress[topic]["correct"] += 1
    save_json(PROGRESS_FILE, progress)

def show_progress():
    progress = load_json(PROGRESS_FILE, {})
    if not progress:
        print("No progress tracked yet.")
        return
    for topic, stats in progress.items():
        print(f"{topic}: {stats['correct']}/{stats['answered']} correct")

# --- LangChain Q&A ---
def llm_agent_session(topic):
    llm = OpenAI(temperature=0.7)
    print(f"\nExploring topic: {topic}")
    # Step 1: Generate a set of questions to assess knowledge
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Generate 5 questions of increasing difficulty to assess a student's knowledge of {topic}. Return as a numbered list."
    )
    questions = llm(prompt.format(topic=topic))
    questions_list = [q.strip() for q in questions.split('\n') if q.strip() and q[0].isdigit()]
    if not questions_list:
        print("Could not generate questions. Using fallback.")
        questions_list = [f"What is {topic}?", f"Explain a key concept in {topic}."]
    correct_count = 0
    for i, question in enumerate(questions_list, 1):
        print(f"\nQuestion {i}: {question}")
        answer = input("Your answer: ")
        # Use LLM to check answer correctness
        check_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="Question: {question}\nStudent Answer: {answer}\nIs this answer correct? Reply only 'yes' or 'no'."
        )
        result = llm(check_prompt.format(question=question, answer=answer)).strip().lower()
        if result.startswith('yes'):
            print("Correct!")
            correct = True
            correct_count += 1
        else:
            print("Not quite. Keep practicing!")
            correct = False
        track_progress(topic, correct)
    print(f"\nAssessment complete. You answered {correct_count}/{len(questions_list)} correctly.")
    # Step 2: Start training based on performance
    if correct_count < len(questions_list) // 2:
        print("Let's start with the basics.")
        training_level = "beginner"
    else:
        print("You seem to know the basics. Advancing to intermediate training.")
        training_level = "intermediate"
    # Training loop (ask 3 more questions)
    for i in range(3):
        train_prompt = PromptTemplate(
            input_variables=["topic", "level"],
            template="Ask a {level} training question about {topic}."
        )
        train_q = llm(train_prompt.format(topic=topic, level=training_level))
        print(f"\nTraining Question: {train_q}")
        answer = input("Your answer: ")
        check_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="Question: {question}\nStudent Answer: {answer}\nIs this answer correct? Reply only 'yes' or 'no'."
        )
        result = llm(check_prompt.format(question=train_q, answer=answer)).strip().lower()
        if result.startswith('yes'):
            print("Correct!")
            track_progress(topic, True)
        else:
            print("Not quite. Keep practicing!")
            track_progress(topic, False)
    print("\nTraining session complete!")

def interactive_session():
    while True:
        print("\n--- Main Menu ---")
        print("1. List topics")
        print("2. Add topic")
        print("3. Delete topic")
        print("4. Show progress")
        print("5. Study a topic")
        print("0. Exit")
        choice = input("Choose an option: ")
        if choice == '1':
            list_topics()
        elif choice == '2':
            add_topic()
        elif choice == '3':
            delete_topic()
        elif choice == '4':
            show_progress()
        elif choice == '5':
            topics = list_topics()
            if topics:
                idx = input("Enter topic number to study: ")
                try:
                    idx = int(idx) - 1
                    if 0 <= idx < len(topics):
                        llm_agent_session(topics[idx])
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    interactive_session()
