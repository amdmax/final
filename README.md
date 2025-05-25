# Interactive Learning System

This project is a full-stack adaptive learning platform designed to deliver personalized, agentic learning experiences. It features a Python FastAPI backend (leveraging LangChain for LLM-driven logic) and a modern React + TypeScript frontend (Vite).

## Project Overview

The system enables students to select or add learning topics, answer dynamically generated questions, and receive instant, LLM-evaluated feedback and learning tips. Each topic is managed per student using unique GUIDs, and all progress, questions, and interactions are tracked and persisted for a truly adaptive experience.

## Goals

- **Personalized Learning:** Deliver a tailored learning journey for each student, adapting questions and feedback based on their progress and responses.
- **Agentic Interactions:** Implement an agent that not only quizzes but also evaluates, encourages, and guides the student, providing actionable learning tips and resources.
- **Modern UI/UX:** Provide a clean, intuitive, and responsive interface for topic management, study sessions, and feedback display.
- **Robust Data Model:** Store all topics, questions, progress, and interaction history per student/topic, supporting multi-student scenarios and persistent memory.
- **LLM Integration:** Use advanced LLMs (via LangChain) for question generation, answer evaluation (scoring 1-10), and feedback/tip creation.

## Challenges Faced

- **Per-Student/Topic Data Management:** Refactoring the backend to support GUID-based topic management and per-student state required careful migration and robust helper functions.
- **LLM Response Parsing:** LLMs often return responses in varied formats (JSON, plain text, or wrapped in metadata). Extracting clean feedback and tips for the UI required advanced parsing and fallback logic.
- **Agentic Workflow:** Designing an agent that can adaptively interact, assess, and encourage students—while persisting all relevant state and memory—was a significant architectural challenge.
- **Frontend Synchronization:** Ensuring the React frontend always reflects the latest backend data model and API, especially as endpoints and data structures evolved.
- **UI/UX Consistency:** Achieving a modern, accessible, and visually consistent interface across topic management, study, and feedback panels, including responsive design for different devices.

## Project Structure

- `backend/` — FastAPI backend with LangChain logic
- `frontend/` — React + TypeScript frontend (Vite)

## Backend Setup

1. Navigate to the backend directory:
   ```zsh
   cd backend
   ```
2. Activate the virtual environment:
   ```zsh
   source .venv/bin/activate
   ```
3. Run the FastAPI server:
   ```zsh
   uvicorn api:app --reload
   ```

## Frontend Setup

1. Navigate to the frontend directory:
   ```zsh
   cd frontend
   ```
2. Start the development server:
   ```zsh
   npm run dev
   ```

## API Documentation
- The backend exposes OpenAPI docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---
