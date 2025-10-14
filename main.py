import pandas as pd
from modules.data_preprocessing import load_datasets, create_question_with_prompt
from modules.ai_prompting import generate_ai_answers
from modules.ai_detector import run_ai_detector

DATASET_NAME = "writingprompts_QA.parquet"
PROMPT_FILE = "prompt.csv"

AI_MODEL_NAME = "llama"
AI_DETECTOR_NAME = "roberta"

def main():
    print("=== Loading data ===")
    df, prompt = load_datasets(DATASET_NAME,PROMPT_FILE)
    
    if True:
        # randomly pick some rows
        df = df.sample(n=2, random_state=69) # nice

    print("=== Generating AI answers ===")
    df = generate_ai_answers(df, AI_MODEL_NAME, "question")
    
    print("=== Running AI detector ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "answer")
    
    print("=== Running AI detector ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "question_answer_ai")

    print("=== Creating Question with Prompt ===")
    df = create_question_with_prompt(df, prompt)
    
    print("=== Generating AI answer for question with prompt ===")
    df = generate_ai_answers(df, AI_MODEL_NAME, "question_with_prompt")
    
    print("=== Running AI detector for answer to question with prompt ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "question_with_prompt_answer_ai")
    
    print("=== Creating CSV output file ===")
    df.to_csv('output.csv', index = False)
    
    print("=== Finished ===")
    

if __name__ == "__main__":
    main()
