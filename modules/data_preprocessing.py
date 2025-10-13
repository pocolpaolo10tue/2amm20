import pandas as pd
import csv

def load_datasets(dataset_name, prompt_name):
    # Try to automatically detect delimiter
    with open(f"datasets/{prompt_name}", 'r', encoding='latin1') as f:
        sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter

    print(f"Detected delimiter: {repr(delimiter)}")

    prompts = pd.read_csv(
        f"datasets/{prompt_name}",
        encoding="latin1",      # latin1 works on both Windows/Linux for ANSI
        sep=delimiter,          # auto-detected delimiter
        engine="python",        # more tolerant
        on_bad_lines="skip"     # ignore malformed lines
    )

    # Load dataset (assuming your original code)
    df = pd.read_csv(f"datasets/{dataset_name}", encoding="utf-8")

    return df, prompts

def data_cleaning(dataset_name, df):
    if dataset_name == "writingprompts_QA.parquet":
        if "question" in df.columns:
            df["question"] = df["question"].apply(
                lambda x: x[len("[ WP ]"):].strip() if isinstance(x, str) and x.startswith("[ WP ]") else x
            )
    return df
    
def create_question_with_prompt(questions: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    questions["key"] = 1
    prompts["key"] = 1
    combined = pd.merge(questions, prompts, on="key").drop("key", axis=1)
    combined["question_with_prompt"] = combined["question"] + " " + combined["prompt"]
    return combined

