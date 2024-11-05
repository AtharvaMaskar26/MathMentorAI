import os 
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv()

'''
TODO Agents: 
1. Question checker - Compare answers and if the answer is correct return True or return False
2. Student Confusion detection - Detect confusion in student's response
'''

answer_checker_agent = Agent(
    role="7th grade answer checker supervisor", 
    goal="Given the correct answer and a student's answer for a given question. Check if the answer given by the student is correct or not.", 
    backstory="You are a answer checker supervisor, who correct's student's answer for a particular question. You have understand the approach the student has used instead of just checking the answer."
)

student_confusion_detection_agent = Agent(
    role="7th grade student confusion detector.", 
    goal="Given the answer to a particular question by a student, you have to analyzethe answer and undetstand where the confusion is.", 
    backstory="You love to help students out with their questions and undetstand where they lack"
)

def check_student_answer(student_answer: str, original_question: str, ground_turth: str) -> bool:
    task1 = Task(
        description=f"For the given question: {original_question} and the correct answer: {ground_turth}. Check if the student has given the correct answer: {student_answer}", 
        expected_output="return True if the answer is correct else return False. Only return the output in boolean and no extra text.", 
        agent=answer_checker_agent
    )

    student_checker_crew = Crew(
        agents=[answer_checker_agent], 
        tasks=[task1], 
        verbose=True
    )

    result = student_checker_crew.kickoff()
    result = result.raw

    return bool(result)

def analyze_student_confusion(student_answer: str, original_question: str, ground_turth: str) -> str:
    task2 = Task(
        description=f"For the given question: {original_question} and correct expected answer: {ground_turth}. The student gave the following answer: {student_answer}. Analyze the answer and detect what the student's confusion was.", 
        expected_output="Just return the student's confusion in 1 line, without any extra text.", 
        agent=check_student_answer
    )

    student_confusion_analyze_crew = Crew(
        agents=[student_confusion_detection_agent], 
        tasks=[task2], 
        verbose=True
    )

    result = student_confusion_analyze_crew.kickoff()

    return result