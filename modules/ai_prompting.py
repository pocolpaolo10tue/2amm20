import pandas as pd
from llamainteract.llamapythonapi import LlamaInterface
import inspect

def generate_ai_answers(df, model_name, question_column):
    if model_name == "llama":
        return run_llama(df, question_column)


def run_llama(df, question_column):
    """
        This function takes as input a dataframe df and a question_column (string).
        The dataframe df contains a column with the name question_column. Each entry in this column is a question (string).
        The function uses the Llama model to generate an answer each question in question_column.
        The function returns the dataframe df with an additional column "answer" containing the generated answers.
    """

    llama = LlamaInterface()
    print('FLorian Output code:::')
    print(print(llama._class.module_))
    print(inspect.getsource(LlamaInterface))
    df[question_column + "_answer_ai"] = df[question_column].apply(
        lambda x: llama.qa(x)["choices"][0]["text"].strip()
    )
    
    return df