import pandas as pd
import multiprocessing as mp
from llamainteract.llamapythonapi import LlamaInterface
import os
import time

def generate_ai_answers(df, model_name, question_column, n_workers=None, chunk_size=256):
    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    if model_name == "llama":
        return run_llama(df, question_column, n_workers=n_workers, chunk_size=chunk_size)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

def worker_fn(questions_chunk):
    """
    Worker process: initializes its own LlamaInterface and answers a chunk.
    Each worker runs in its own process (isolated from others).
    """
    start_total = time.time()
    
    # Measure model instantiation
    start_init = time.time()
    llama = LlamaInterface()
    end_init = time.time()
    print(f"[Worker] LlamaInterface initialization took {end_init - start_init:.2f} seconds")

    # Measure batch inference
    start_infer = time.time()
    answers = llama.batch_qa(questions_chunk)
    end_infer = time.time()
    print(f"[Worker] batch_qa on {len(questions_chunk)} questions took {end_infer - start_infer:.2f} seconds")

    end_total = time.time()
    print(f"[Worker] Total worker time: {end_total - start_total:.2f} seconds")

    return answers


def parallel_run(questions, n_workers=4, chunk_size=256):
    """
    Run llama inference in parallel across CPU/GPU processes.
    Each worker gets its own model instance.
    """
    chunks = [questions[i:i + chunk_size] for i in range(0, len(questions), chunk_size)]
    results = []

    with mp.Pool(processes=n_workers) as pool:
        for ans_chunk in pool.imap(worker_fn, chunks):
            results.extend(ans_chunk)

    return results


def run_llama(df, question_column, n_workers=4, chunk_size=256):
    """
    Run the llama model on a dataframe column in parallel.
    """
    questions = df[question_column].tolist()

    # Use multiprocessing parallel inference
    answers = parallel_run(questions, n_workers=n_workers, chunk_size=chunk_size)

    # Extract the answers and add them to the DataFrame
    df[question_column + "_answer_ai"] = [ans["choices"][0]["text"].strip() for ans in answers]
    return df

