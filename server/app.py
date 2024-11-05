import streamlit as st
import jax
import jax.numpy as jnp
import json
import jax.random as jrandom
from utils import *
import time

# Constants
NUM_CONTEXTS = 3  # Example number of contexts
NUM_ARMS = 35      # Example number of arms (questions)
EXPLORATION_RATE = 0.1  # Epsilon for the epsilon-greedy strategy

# Function to select a question using epsilon-greedy strategy
@jax.jit
def select_question(rng_key, Q_values, context_index):
    # Generate a random number for exploration decision
    random_value = jrandom.uniform(rng_key)

    # Create a boolean mask for exploration
    explore = random_value < EXPLORATION_RATE

    # Select arm index based on exploration or exploitation
    arm_index = jnp.where(explore, 
                          jrandom.randint(rng_key, (1,), 0, NUM_ARMS)[0],  # Exploration
                          jnp.argmax(Q_values[context_index]))  # Exploitation

    return arm_index
# Custom reward function based on user response and context
@jax.jit
def custom_reward(is_correct, hints_used, timestamp_since_last, correct_ratio):
    # Calculate base reward if the answer is correct
    base_reward = jnp.where(is_correct, 1.0, 0.0)
    
    # Apply hint penalty if the answer is correct
    hint_penalty = 0.2 * hints_used
    reward = base_reward - hint_penalty
    
    # Apply additional penalty for quick responses
    time_penalty = jnp.where(timestamp_since_last < 10, 0.5, 0.0)
    reward -= time_penalty
    
    # Add bonus based on correct ratio if the answer is correct
    reward = jnp.where(is_correct, reward + 0.5 * correct_ratio, reward)

    # Ensure reward is non-negative
    reward = jnp.maximum(reward, 0.0)
    
    return reward


# Function to update Q-values based on received reward
@jax.jit
def update_q_values(Q_values, counts, context_index, arm_index, reward):
    counts = counts.at[context_index, arm_index].add(1)
    Q_values = Q_values.at[context_index, arm_index].add((reward - Q_values[context_index, arm_index]) / counts[context_index, arm_index])
    return Q_values, counts

# Function to display the problems page
def problems_page():
    with open("./user_data.json", "r") as j:
        user_data = json.load(j)
    with open("./data/question_answer_pairs.json", "r") as jsonFile:
        question_answer_data = json.load(jsonFile)

    user_record = next(user for user in user_data if user['username'] == st.session_state.user_id)

    st.subheader("Problems Page")
    # You can add more functionality for the problems page here.
    st.write("Welcome to the Problems Page!")
    # Example of displaying problems (replace with your own logic)

    session_Q_values = st.session_state.Q_values
    context_index = 0
    session_count_values = st.session_state.counts 
    rng_key = jrandom.PRNGKey(0)

    context = int(user_record['context']['correct_attempts'] * (12000 - 1))

    question_index = select_question(rng_key, session_Q_values, context_index)

    # Choosing a random question
    rng_key = jax.random.PRNGKey(st.session_state.current_question_index)
    random_question_index = jax.random.randint(rng_key, (1,), 0, len(question_answer_data[question_index]['question_answer_pairs']))

    random_question_index = random_question_index.item()

    random_question = question_answer_data[question_index]['question_answer_pairs'][random_question_index]

    st.write(random_question['question'])

    with st.form('my-form'):
        user_answer = st.text_input("Enter your answer")

        submit_button = st.form_submit_button("Submit")

        if submit_button:
            current_time = time.time()  # Get the current time

            # Subtract the previously stored timestamp (which should be updated after a correct answer) from the current time
            user_record['context']['timestamp_since_last'] = current_time - user_record['context']['timestamp_since_last']

            is_user_answer_correct = check_student_answer(user_answer, random_question['question'], random_question['answer'])

            print("as", is_user_answer_correct)
            if is_user_answer_correct:
                user_record['context']['total_attempts'] += 1
                user_record['context']['correct_attempts'] += 1
                user_record['context']['correct_ration'] = user_record['context']['correct_attempts'] /  user_record['context']['total_attempts'] 

                # Reward
                # Calculate the reward based on user's response
                reward = custom_reward(1.0, 0, user_record['context']['timestamp_since_last'], user_record['context']['correct_ration'])

                # Updating Q_values
                # global Q_values, counts
                Q_values, counts = update_q_values(session_Q_values, session_count_values, context, question_index, reward)

                user_record['Q_values'] = Q_values
                user_record['counts'] = counts

                st.success("Good job correct answer")
                st.session_state.current_question_index += 1  # Increment question index or handle differently
                st.rerun()  # Rerun the script to show a new 
            else:
                user_record['total_attempts'] += 1

                user_record['correct_ration'] = user_record['context']['correct_attempts'] /  user_record['context']['total_attempts'] 

                # Reward Function
                reward = custom_reward(0.0, 0, user_record['context']['timestamp_since_last'], user_record['context']['correct_ration'])

                # Updating Q_values
                Q_values, counts = update_q_values(session_Q_values, session_count_values, context, question_index, reward)

                user_record['Q_values'] = Q_values
                user_record['counts'] = counts
                st.danger(f"Wrong answer.")

        

# Function to display the signup form
def signup():
    st.subheader("Signup")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Sign Up"):
        if password == confirm_password:
            user_entry = {
                "username": username, 
                "password": password, 
                "Q_values": jnp.zeros((NUM_CONTEXTS, NUM_ARMS)).tolist(),  # Convert to list
                "counts": jnp.zeros((NUM_CONTEXTS, NUM_ARMS)).tolist(),  # Convert to list
                "context": {
                    "timestamp_since_last": 0, 
                    "correct_ratio": 0, 
                    "total_attempts": 0, 
                    "correct_attempts": 0
                }
            }

            with open("./user_data.json", "r") as j:
                d = json.load(j)
            d.append(user_entry)
            with open("./user_data.json", "w") as jsonFile:
                json.dump(d, jsonFile, indent=4)
            st.success(f"Account created for {username}!")
        else:
            st.error("Passwords do not match!")

# Function to display the login form
def login():
    st.session_state.current_question_index = 0
    st.subheader("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    with open("./user_data.json", "r") as jsonFile:
        user_data = json.load(jsonFile)
    
    if st.button("Login"):
        # Check if user exists
        user_found = any(user['username'] == username for user in user_data)
        if user_found:
            user_record = next(user for user in user_data if user['username'] == username)
            if user_record['password'] == password:  # Add password verification
                # Load user data and store in session state
                st.session_state.user_id = username  # Store the logged-in user
                st.session_state.Q_values = jnp.array(user_record['Q_values'])  # Store Q-values
                st.session_state.counts = jnp.array(user_record['counts'])  # Store counts
                st.session_state.context = user_record['context']  # Store context
                st.session_state.logged_in = True  # Set logged_in flag
    

                st.success("Login successful! Redirecting to problems page...")
                st.rerun()  # Refresh the app to show problems page
            else:
                st.error("Invalid password!")
        else:
            st.error("Username not found!")

# Main function to run the app
def main():
    st.title("Simple Login & Signup Page")
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False  # Initialize logged_in flag if not set

    if st.session_state.logged_in:
        problems_page()  # Show problems page if logged in
    else:
        menu = ["Login", "Signup"]
        choice = st.sidebar.selectbox("Select Action", menu)
        
        if choice == "Login":
            login()
        elif choice == "Signup":
            signup()

if __name__ == "__main__":
    main()
