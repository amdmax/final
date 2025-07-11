import { useEffect, useState } from "react";
import "./App.css";

const API_BASE = "/api";
const DEFAULT_STUDENT_ID = "default"; // You may want to make this dynamic later

interface Topic {
  id: string;
  name: string;
}

interface AgenticResponse {
  action: string;
  question?: string;
  message?: string;
  level?: number;
  score?: number;
  last_score?: number;
  last_feedback?: string;
  learning_tip?: string;
  questions?: string[]; // Added to handle multiple questions
}

function App() {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [agentState, setAgentState] = useState<AgenticResponse | null>(null);
  const [userAnswer, setUserAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch topics for the student
  useEffect(() => {
    fetch(`${API_BASE}/students/${DEFAULT_STUDENT_ID}/topics`)
      .then((res) => res.json())
      .then((data) => setTopics(data))
      .catch(() => setTopics([]));
  }, []);

  // Start agentic session when topic is selected
  const startTopic = (topic: Topic) => {
    setSelectedTopic(topic);
    setAgentState(null);
    setUserAnswer("");
    setError(null);
    setLoading(true);
    fetch(`${API_BASE}/students/${DEFAULT_STUDENT_ID}/topics/${topic.id}/interact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic: topic.id, user_id: DEFAULT_STUDENT_ID }),
    })
      .then((res) => res.json())
      .then((data) => {
        // If backend returns {questions: [...], question: ...}, resume from last unanswered
        if (Array.isArray(data.questions) && data.question) {
          setAgentState({
            action: "ask_question",
            question: data.question,
            questions: data.questions,
            message: undefined,
            level: undefined,
            score: undefined,
            last_score: undefined,
            last_feedback: undefined,
            learning_tip: undefined,
          });
        } else {
          setAgentState(data);
        }
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to start topic session.");
        setLoading(false);
      });
  };

  // Submit answer to agentic endpoint
  const submitAnswer = () => {
    if (!selectedTopic || !agentState?.question) return;
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/students/${DEFAULT_STUDENT_ID}/topics/${selectedTopic.id}/interact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic: selectedTopic.id,
        user_id: DEFAULT_STUDENT_ID,
        question: agentState.question,
        answer: userAnswer,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        // If backend returns a new question, keep the feedback and move to next question
        if (data && data.action === "ask_question" && data.question) {
          setAgentState((prev) => ({
            ...data,
            // Optionally, keep a history of questions/answers if desired
          }));
        } else {
          setAgentState(data);
        }
        setUserAnswer("");
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to submit answer.");
        setLoading(false);
      });
  };

  // Add a new topic
  const [newTopicName, setNewTopicName] = useState("");
  const addTopic = () => {
    if (!newTopicName.trim()) return;
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/students/${DEFAULT_STUDENT_ID}/topics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newTopicName }),
    })
      .then((res) => res.json())
      .then((data) => {
        setTopics((prev) => [...prev, data]);
        setNewTopicName("");
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to add topic.");
        setLoading(false);
      });
  };

  // Delete a topic
  const deleteTopic = (topic: Topic) => {
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/students/${DEFAULT_STUDENT_ID}/topics/${topic.id}`, {
      method: "DELETE",
    })
      .then((res) => res.json())
      .then(() => {
        setTopics((prev) => prev.filter((t) => t.id !== topic.id));
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to delete topic.");
        setLoading(false);
      });
  };

  // UI rendering
  return (
    <div className="app-layout">
      <div className="topics-section">
        <h2>Your Topics</h2>
        <div className="topics-list">
          {topics.map((topic) => (
            <div key={topic.id} style={{ display: 'inline-flex', alignItems: 'center', marginRight: 8 }}>
              <button
                className={`topic-btn${selectedTopic?.id === topic.id ? ' selected' : ''}`}
                onClick={() => startTopic(topic)}
                disabled={loading}
              >
                {topic.name}
              </button>
              <button
                aria-label={`Delete ${topic.name}`}
                style={{ marginLeft: 4, color: 'red', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 'bold', fontSize: 18 }}
                onClick={() => deleteTopic(topic)}
                disabled={loading}
                title="Remove topic"
              >
                ×
              </button>
            </div>
          ))}
        </div>
        <div className="add-topic-form">
          <input
            type="text"
            placeholder="Add new topic..."
            value={newTopicName}
            onChange={(e) => setNewTopicName(e.target.value)}
            disabled={loading}
          />
          <button onClick={addTopic} disabled={loading || !newTopicName.trim()}>
            Add Topic
          </button>
        </div>
      </div>
      <div className="study-section">
        {selectedTopic && (
          <>
            <h2>Studying: {selectedTopic.name}</h2>
            {agentState && (
              <div className="agentic-panel">
                {agentState.action === "ask_question" && (
                  <>
                    {agentState.last_feedback && (
                      <div className="feedback-block">
                        <strong>Feedback for the last question:</strong>
                        <pre className="feedback-text">{agentState.last_feedback}</pre>
                      </div>
                    )}
                    {agentState.learning_tip && (
                      <div className="tip-block">
                        <strong>Learning Tip:</strong>
                        <div className="tip-text">{agentState.learning_tip}</div>
                      </div>
                    )}
                    {(agentState.last_feedback || agentState.learning_tip) && (
                      <div className="section-divider"></div>
                    )}
                    <div className="question-block">
                      <strong>Question:</strong>
                      <div className="question-text">{agentState.question}</div>
                    </div>
                    <div className="answer-form">
                      <input
                        type="text"
                        placeholder="Your answer..."
                        value={userAnswer}
                        onChange={(e) => setUserAnswer(e.target.value)}
                        disabled={loading}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") submitAnswer();
                        }}
                      />
                      <button onClick={submitAnswer} disabled={loading || !userAnswer.trim()}>
                        Submit
                      </button>
                    </div>
                  </>
                )}
                {agentState.action === "level_up" && (
                  <div className="levelup-block">
                    <strong>{agentState.message}</strong>
                    <div>Score: {agentState.score}</div>
                    <button onClick={() => startTopic(selectedTopic!)} disabled={loading}>
                      Continue
                    </button>
                  </div>
                )}
                {agentState.action === "complete" && (
                  <div className="complete-block">
                    <strong>{agentState.message}</strong>
                    {agentState.learning_tip && (
                      <div className="tip-block">
                        <strong>Final Learning Tip:</strong>
                        <div className="tip-text">{agentState.learning_tip}</div>
                      </div>
                    )}
                    <button
                      className="back-to-topics-btn"
                      onClick={() => setSelectedTopic(null)}
                      disabled={loading}
                    >
                      Back to Topics
                    </button>
                  </div>
                )}
                {agentState && Array.isArray(agentState.questions) && (
                  <div className="questions-list">
                    <strong>All Questions:</strong>
                    <ol>
                      {agentState.questions.map((q, idx) => (
                        <li key={idx}>{q}</li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}
            {!agentState && <div>Loading session...</div>}
          </>
        )}
      </div>
      {error && <div className="error-block">{error}</div>}
    </div>
  );
}

export default App;
