# Math Mentor AI 📐💡

![Math Mentor AI]

### Your Personal AI-Powered Math Tutor

Math Mentor AI is an interactive, personalized AI-driven math tutor designed to support students in mastering math concepts at their own pace. By leveraging the **LLaMA 3.2 3B** model fine-tuned on the **MathDial** dataset, Math Mentor AI enhances the math learning experience with dialogue-based problem-solving and guidance. This project uses **Streamlit** to create a user-friendly interface for easy interaction with the AI tutor.

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [How It Works](#how-it-works)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **AI-Powered Math Tutoring**: Engage with a responsive AI tutor that adapts to each student’s level and provides targeted support for complex math problems.
- **Dialogue-Based Problem Solving**: Fine-tuned on the MathDial dataset, Math Mentor AI improves its conversation flow, making problem-solving an interactive experience.
- **Personalized Learning Path**: Using the Contextual Bandits algorithm, Math Mentor AI tailors hints, guidance, and challenges based on user interaction history, helping students progress effectively.
- **Interactive UI with Streamlit**: A simple, web-based interface that allows students to input questions, get responses, and track their progress.

## Getting Started

### Installation

To run Math Mentor AI locally with Streamlit, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AtharvaMaskar26/MathMentorAI.git
   cd Math-Mentor-AI
   ```

2. **Install Dependencies**
   - We recommend creating a virtual environment first:
     ```bash
     python -m venv env
     source env/bin/activate  # On Windows, use `env\Scripts\activate`
     ```
   - Then install required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download Model Weights**
   - Download the LLaMA 3.2 3B model weights and place them in the `models` directory. (Ensure you have the appropriate permissions to use the weights.)

### Usage

To start the Math Mentor AI app using Streamlit:

1. **Run the Streamlit Application**  
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**  
   - Open your web browser and go to `http://localhost:8501` to interact with Math Mentor AI.
   - Type in math questions or problems, and Math Mentor AI will provide personalized explanations and guidance.

---

## How It Works

Math Mentor AI leverages the LLaMA 3.2 3B model, fine-tuned specifically on the **MathDial dataset**, to create a conversational tutor that guides students through math challenges. The **Contextual Bandits algorithm** helps tailor the experience based on individual learning patterns, making each session unique and aligned with the student’s progress. The **Streamlit** interface provides an intuitive and user-friendly way for students to interact with the AI.

---

## Future Plans

We are continually working to enhance Math Mentor AI and have some exciting features planned:

1. **Real-Time AI Tutor**: Implement real-time assistance where students can interact with the tutor in live problem-solving sessions.
2. **Full-Stack Implementation**: Expand Math Mentor AI into a full-stack application with user profiles, progress tracking, and a backend server.

---

## Contributing

We welcome contributions! To contribute, please:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m "Add feature-name"`).
5. Push to the branch (`git push origin feature-name`).
6. Create a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> **Note:** Math Mentor AI is in the early stages of development. We appreciate all feedback and suggestions for future improvements!

--- 

This setup makes your project easier to use and invites contributions. Let me know if you'd like more sections or details added!