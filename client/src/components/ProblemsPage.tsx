import React, { useState, useEffect } from 'react'
import axios from 'axios'

interface ProblemsPageProps {
  username: string
}

interface Question {
  id: number
  question: string
}

export default function ProblemsPage({ username }: ProblemsPageProps) {
  const [question, setQuestion] = useState<Question | null>(null)
  const [answer, setAnswer] = useState('')
  const [feedback, setFeedback] = useState('')

  useEffect(() => {
    fetchQuestion()
  }, [])

  const fetchQuestion = async () => {
    try {
      const response = await axios.get('/api/question', { params: { username } })
      setQuestion(response.data.question)
    } catch (error) {
      console.error('Error fetching question:', error)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const response = await axios.post('/api/answer', { username, questionId: question?.id, answer })
      setFeedback(response.data.feedback)
      if (response.data.correct) {
        setTimeout(() => {
          setFeedback('')
          setAnswer('')
          fetchQuestion()
        }, 2000)
      }
    } catch (error) {
      console.error('Error submitting answer:', error)
    }
  }

  return (
    <div className="max-w-2xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Problem Solving</h2>
      {question ? (
        <div className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
          <h3 className="text-lg font-semibold mb-4">{question.question}</h3>
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="answer">
                Your Answer
              </label>
              <input
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                id="answer"
                type="text"
                placeholder="Enter your answer"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
              />
            </div>
            <div className="flex items-center justify-between">
              <button
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                type="submit"
              >
                Submit
              </button>
            </div>
          </form>
          {feedback && (
            <div className={`mt-4 p-2 rounded ${feedback.includes('Correct') ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
              {feedback}
            </div>
          )}
        </div>
      ) : (
        <p>Loading question...</p>
      )}
    </div>
  )
}