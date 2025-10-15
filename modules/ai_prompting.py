import pandas as pd
from llamainteract.llamapythonapi import LlamaInterface

def generate_ai_answers(df, model_name, question_column):
    if model_name == "llama":
        return run_llama(df, question_column)

_llama_cache = {}

def run_llama(df, question_column):
    global _llama_cache
    if "llama" not in _llama_cache:
        _llama_cache["llama"] = LlamaInterface()

    llama = _llama_cache["llama"]
    questions = df[question_column].tolist()
    answers = llama.batch_qa(questions)

    # Extract the answers and add them to the DataFrame
    df[question_column + "_answer_ai"] = [ans["choices"][0]["text"].strip() for ans in answers]
    return df

