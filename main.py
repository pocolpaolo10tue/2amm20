import pandas as pd
from modules.data_preprocessing import load_datasets, create_question_with_prompt, clean_dataset
from modules.ai_prompting import generate_ai_answers
from modules.ai_detector import run_ai_detector
import multiprocessing as mp

DATASET_NAME = "stackexchange_QA.parquet"
PROMPT_FILE = "overall_prompt.csv"

AI_MODEL_NAME = "llama"
AI_DETECTOR_NAME = "detectgpt"

NUMBER_OF_QUESTIONS = 100
MIN_LENGTH_ANSWER = 100
MAX_LENGTH_QUESTION = 1000

def main():
    print("=== Loading data ===")
    df, prompt = load_datasets(DATASET_NAME,PROMPT_FILE)
   
    print("=== Clean and limit the dataset ===")
    df = clean_dataset(df, NUMBER_OF_QUESTIONS, MIN_LENGTH_ANSWER, MAX_LENGTH_QUESTION)

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
    df.to_csv(f"output_{NUMBER_OF_QUESTIONS}_{AI_DETECTOR_NAME}_{PROMPT_FILE}", index = False)
    
    print("=== Finished ===")

# Tested Parameters, it's unnecessary to test the default values for everything if it's covered by the baseline
PARAM_GRID = [
    # Baseline
    {},
    
    #Temperature
    {"temperature": 0.0},
    {"temperature": 0.2},
    {"temperature": 0.5},
    # {"temperature": 0.8},
    {"temperature": 1.0},
    {"temperature": 1.4},
    {"temperature": 2.0},

    # Top-p
    {"top_p": 0.6},
    {"top_p": 0.9},
    # {"top_p": 0.95},
    {"top_p": 1.0},
    
    # # Top-k
    {"top_k": 5},
    {"top_k": 10},
    # {"top_k": 40},
    {"top_k": 100},
    
    # Repeat Penalty
    {"repeat_penalty": 1.0},
    # {"repeat_penalty": 1.1},
    {"repeat_penalty": 1.2},
    {"repeat_penalty": 1.5}
]

# Define default values
DEFAULT_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1
}

def main_params_test():
    print("=== Loading data ===")
    df, prompt = load_datasets(DATASET_NAME, PROMPT_FILE)
   
    print("=== Clean and limit the dataset ===")
    df = clean_dataset(df, NUMBER_OF_QUESTIONS, MIN_LENGTH_ANSWER, MAX_LENGTH_QUESTION)

    print("=== Starting parameter sweep ===")
    # Collect results from all parameter sets into a single CSV
    all_results = []

    for i, param_set in enumerate(PARAM_GRID, 1):
        # Merge operator on the datasets, param set overrides defaults if specified
        full_params = {**DEFAULT_PARAMS, **param_set}
        df_results = generate_ai_answers(
            df.copy(),
            model_name=AI_MODEL_NAME,
            question_column="question",
            temperature=full_params["temperature"],
            top_p=full_params["top_p"],
            top_k=full_params["top_k"],
            repeat_penalty=full_params["repeat_penalty"]
        )
        df_results = run_ai_detector(AI_DETECTOR_NAME, df_results, "answer")

        # Add parameter info to each row for later analysis
        for key, val in full_params.items():
            df_results[key] = val
        all_results.append(df_results)

    print("=== Creating CSV output file ===")
    df_combined = pd.concat(all_results, ignore_index=True)
    output_name = f"output_param_{NUMBER_OF_QUESTIONS}_{AI_DETECTOR_NAME}_{PROMPT_FILE}.csv"
    df_combined.to_csv(output_name, index=False)

    print(f"=== Finished ===")

if __name__ == "__main__":
    main()
