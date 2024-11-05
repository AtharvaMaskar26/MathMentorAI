import streamlit as st
import pandas as pd
import numpy as np
from jax import random, jit
import jax.numpy as jnp
from datetime import datetime

# Constants
n_arms = 15  # Number of problem types
context_dim = 4  # Number of context features

# Load questions
questions = pd.read_csv("./data/external_df.csv")

# Initialize parameters
weights = jnp.zeros((n_arms, context_dim))
learning_rate = 0.1
gamma = 0.95

# Initialize Q-values and counts
Q_values = jnp.zeros(n_arms)
counts = jnp.zeros(n_arms)

# Store last attempted timestamps for (type, level) combinations
last_attempted_timestamps = {(type_, level): None for type_ in ['Algebra', 'Calculus', 'Pre-calculus'] for level in range(1, 6)}

# Initialize success and failure counts
successes = jnp.zeros(n_arms)
failures = jnp.zeros(n_arms)

@jit
def contextual_bandit_selection(weights, context):
    expected_rewards = jnp.dot(weights, context)
    return jnp.argmax(expected_rewards), expected_rewards

@jit
def update_Q_values(arm_index, reward, Q_values, expected_rewards, gamma, learning_rate):
    best_next_reward = jnp.max(expected_rewards)
    Q_values = Q_values.at[arm_index].set(
        Q_values[arm_index] + learning_rate * (reward + gamma * best_next_reward - Q_values[arm_index])
    )
    return Q_values

def calculate_time_since_last_attempt(problem_type, level):
    last_time = last_attempted_timestamps[(problem_type, level)]
    if last_time is None:
        return 0.0  # If never attempted, use 0
    else:
        time_since_last_attempt = (datetime.now() - last_time).total_seconds()  # in seconds
        return min(time_since_last_attempt / 3600.0, 24.0)  # Normalize to hours (capped at 24 hours)

def run_contextual_bandit(recent_performance, time_of_day, session_count):
    global weights, Q_values
    
    # Calculate time since last attempt for each (type, level) problem
    problem_type = np.random.choice(['Algebra', 'Calculus', 'Pre-calculus'])
    level = np.random.randint(1, 6)
    time_since_last_attempt = calculate_time_since_last_attempt(problem_type, level)

    context = jnp.array([recent_performance, time_of_day, session_count, time_since_last_attempt])
    arm_index, expected_rewards = contextual_bandit_selection(weights, context)

    selected_question = questions[(questions['type'] == problem_type) & (questions['level'] == level)]
    
    if not selected_question.empty:
        question_text = selected_question.iloc[0]['question']
        solution_text = selected_question.iloc[0]['solution']
    else:
        question_text = "No question found."
        solution_text = ""

    # Simulate a response (for demonstration purposes)
    correct = np.random.rand() < 0.5  # Randomly determining correctness for simulation
    reward = 1.0 if correct else 0.0  # Basic reward structure for demonstration

    # Update Q-values and weights
    Q_values = update_Q_values(arm_index, reward, Q_values, expected_rewards, gamma, learning_rate)
    weights = weights.at[arm_index].set(weights[arm_index] + learning_rate * reward * context)

    # Update the timestamp for the problem type and level
    last_attempted_timestamps[(problem_type, level)] = datetime.now()

    return question_text, solution_text, correct

def main():
    st.title("Math Mentor AI")
    
    # User inputs
    user_id = st.text_input("Enter your User ID:")
    recent_performance = st.slider("Recent Performance (0 to 1):", 0.0, 1.0, 0.5)
    time_of_day = st.selectbox("Time of Day:", [0.0, 1.0])  # 0.0 for day, 1.0 for night
    session_count = st.number_input("Number of Sessions:", 1, 10, 1)

    if st.button("Get Question"):
        question, solution, correct = run_contextual_bandit(recent_performance, time_of_day, session_count)
        st.write(f"Question: {question}")
        if correct is not None:
            st.write(f"Your answer is {'correct' if correct else 'incorrect'}.")
        st.write(f"Solution: {solution}")

if __name__ == "__main__":
    main()
