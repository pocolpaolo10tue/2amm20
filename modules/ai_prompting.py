import pandas as pd
from llamainteract.llamapythonapi import LlamaInterface
import time

# Default values provided from the llama-cpp specification
def generate_ai_answers(df, model_name, question_column, chunk_size=256, temperature=0.8, top_p=0.95, top_k=40, repeat_penalty=1.1):
    if model_name == "llama":
        return run_llama(df, question_column, chunk_size=chunk_size, t=temperature, p=top_p, k=top_k, r=repeat_penalty)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

def run_llama(df, question_column, chunk_size, t, p, k, r):
    """
    Run the llama model on a dataframe column (single process version).
    Includes timing for model initialization and inference.
    """
    questions = df[question_column].tolist()

    print(f"[Main] Starting LlamaInterface initialization...")
    start_init = time.time()
    llama = LlamaInterface()
    end_init = time.time()
    print(f"[Main] LlamaInterface initialization took {end_init - start_init:.2f} seconds")

    results = []

    # Process in chunks to simulate previous parallel code
    for i in range(0, len(questions), chunk_size):
        chunk = questions[i:i + chunk_size]
        print(f"[Main] Processing chunk {i // chunk_size + 1} with {len(chunk)} questions...")

        start_infer = time.time()
        answers = llama.batch_qa(chunk, temperature=t, top_p=p, top_k=k, repeat_penalty=r)
        end_infer = time.time()

        print(f"[Main] batch_qa on {len(chunk)} questions took {end_infer - start_infer:.2f} seconds")
        results.extend(answers)

    total_time = end_infer - start_init
    print(f"[Main] Total runtime (init + inference): {total_time:.2f} seconds")

    # Extract answers and add them to DataFrame
    df[question_column + "_answer_ai"] = [ans["choices"][0]["text"].strip() for ans in results]
    return df
