import React, { useState, useEffect } from 'react';

const App = () => {
  const [question, setQuestion] = useState(null);
  const [questionId, setQuestionId] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [recognition, setRecognition] = useState(null);
  const [listening, setListening] = useState(false);
  const [groundTruth, setGroundTruth] = useState(null);
  const [showAnswer, setShowAnswer] = useState(false);

  // Initialize the webkitSpeechRecognition API
  useEffect(() => {
    const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
    if (SpeechRecognition) {
      const rec = new SpeechRecognition();
      rec.continuous = true;
      rec.interimResults = true;

      rec.onresult = (event) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          if (result.isFinal) {
            finalTranscript += result[0].transcript + ' ';
          }
        }
        setTranscript(finalTranscript);
      };

      rec.onend = () => {
        setListening(false);
      };

      setRecognition(rec);
    } else {
      console.error("Speech recognition not supported");
    }
  }, []);

  // Fetch a random question from the FastAPI backend
  const fetchRandomQuestion = async () => {
    try {
      const response = await fetch("http://localhost:8000/question");
      if (!response.ok) {
        throw new Error("Failed to fetch question");
      }
      const data = await response.json();
      setQuestion(data.question);
      setQuestionId(data.id);
      setGroundTruth(data.ground_truth);
      setFeedback(null);
      setTranscript('');
      setShowAnswer(false);
    } catch (error) {
      console.error("Error fetching question:", error);
      setFeedback("Error fetching question. Please try again later.");
    }
  };

  // Send the user's answer to the backend for validation
  const submitAnswer = async () => {
    if (!questionId || !question) return;

    try {
      const response = await fetch(`http://localhost:8000/answer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_answer: transcript.trim(),
          question_id: questionId,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to submit answer");
      }

      const data = await response.json();
      const isCorrect = data.message;

      if (isCorrect) {
        setFeedback("Correct! Loading the next question...");
        setTimeout(fetchRandomQuestion, 2000);
      } else {
        setFeedback("Incorrect. Try again or click 'Show Answer' to see the solution.");
      }
    } catch (error) {
      console.error("Error submitting answer:", error);
      setFeedback("Error submitting answer. Please try again.");
    }
  };

  const startListening = () => {
    if (recognition) {
      recognition.start();
      setListening(true);
    }
  };

  const stopListening = () => {
    if (recognition) {
      recognition.stop();
    }
  };

  const handleShowAnswer = () => {
    setShowAnswer(true);
  };

  useEffect(() => {
    fetchRandomQuestion();
  }, []);

  return (
    <div className="flex flex-col items-center p-6">
      <h1 className="text-2xl font-bold mb-4">AI Math Tutor</h1>
      {question ? (
        <div className="max-w-2xl w-full">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <p className="text-lg mb-4 font-medium">{question}</p>
            
            <div className="flex flex-wrap gap-2 mb-4">
              <button
                onClick={startListening}
                disabled={listening}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition-colors"
              >
                {listening ? "Listening..." : "Start Answer"}
              </button>
              <button
                onClick={stopListening}
                disabled={!listening}
                className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded transition-colors"
              >
                Stop Listening
              </button>
              <button
                onClick={submitAnswer}
                className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded transition-colors"
              >
                Submit Answer
              </button>
              <button
                onClick={handleShowAnswer}
                className="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded transition-colors"
              >
                Show Answer
              </button>
            </div>

            <div className="mt-4">
              <p className="text-gray-700">
                <strong>Your Answer:</strong> 
                <span className="ml-2">{transcript}</span>
              </p>
              
              {feedback && (
                <p className={`mt-2 text-lg ${
                  feedback.includes("Correct") ? "text-green-600" : "text-red-600"
                }`}>
                  {feedback}
                </p>
              )}
              
              {showAnswer && groundTruth && (
                <div className="mt-4 p-4 bg-gray-50 rounded-md">
                  <p className="text-gray-800">
                    <strong>Correct Answer:</strong>
                    <span className="ml-2">{groundTruth}</span>
                  </p>
                </div>
              )}
            </div>
          </div>

          <button
            onClick={fetchRandomQuestion}
            className="mt-4 bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded transition-colors"
          >
            Next Question
          </button>
        </div>
      ) : (
        <p className="text-gray-600">Loading question...</p>
      )}
    </div>
  );
};

export default App;
