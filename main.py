import pandas as pd
from modules.data_preprocessing import load_datasets, create_question_with_prompt, clean_dataset
from modules.ai_prompting import generate_ai_answers
from modules.ai_detector import run_ai_detector
import multiprocessing as mp

DATASET_NAME = "stackexchange_QA.parquet"
PROMPT_FILE = "prompt.csv"

AI_MODEL_NAME = "llama"
AI_DETECTOR_NAME = "binoculars"

NUMBER_OF_QUESTIONS = 5
MIN_LENGTH_ANSWER = 100

def main():
    mp.set_start_method("spawn", force=True)
    
    print("=== Loading data ===")
    df, prompt = load_datasets(DATASET_NAME,PROMPT_FILE)
   
    print("=== Clean and limit the dataset ===")
    df = clean_dataset(df, NUMBER_OF_QUESTIONS, MIN_LENGTH_ANSWER)

    print("=== Generating AI answers ===")
    df = generate_ai_answers(df, AI_MODEL_NAME, "question")
    
    print("=== Running AI detector on human text ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "answer")
    
    print("=== Running AI detector on AI generated text ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "question_answer_ai")

    print("=== Creating Question with Prompt ===")
    df = create_question_with_prompt(df, prompt)
    
    print("=== Generating AI answer for question with prompt ===")
    df = generate_ai_answers(df, AI_MODEL_NAME, "question_with_prompt")
    
    print("=== Running AI detector for answer to question with prompt ===")
    df = run_ai_detector(AI_DETECTOR_NAME, df, "question_with_prompt_answer_ai")
    
    print("=== Creating CSV output file ===")
    df.to_csv(f"output_{NUMBER_OF_QUESTIONS}_{AI_DETECTOR_NAME}.csv", index = False)
    
    print("=== Finished ===")
    

if __name__ == "__main__":
    main()
