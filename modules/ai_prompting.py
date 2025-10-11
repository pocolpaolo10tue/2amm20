import pandas as pd

def generate_ai_answers(df, model_name, question_name):
    if model_name == "llama":
        return run_llama(df, question_name)


def run_llama(df, question_name):
    # need to generate the answer of the question of question_name and add it to df as question_name + "_answer"
    
    return df


