from binoculars import Binoculars
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np


# --- Dispatcher ---
def run_ai_detector(detector_name, df, answer_name):
    name = detector_name.lower()
    if name == "binoculars":
        return run_binoculars(df, answer_name)
    elif name == "roberta":
        return run_roberta(df, answer_name)
    elif name == "detectgpt":
        return run_detectgpt(df, answer_name)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


# --- Binoculars ---
def run_binoculars(df, answer_name):
    bino = Binoculars()

    scores, preds = [], []
    for text in df[answer_name]:
        scores.append(bino.compute_score(text))
        preds.append(bino.predict(text))

    df[answer_name + "_ai_detection_score"] = scores
    df[answer_name + "_ai_detection_prediction"] = preds
    return df


# --- RoBERTa ---
def run_roberta(df, answer_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[RoBERTa] Using device: {device}")
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    scores, preds = [], []
    for text in df[answer_name]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            ai_prob = probs[0][1].item()
            pred = "AI-generated" if ai_prob > 0.5 else "Human-written"
        scores.append(ai_prob)
        preds.append(pred)

    df[answer_name + "_ai_detection_score"] = scores
    df[answer_name + "_ai_detection_prediction"] = preds
    return df


# --- DetectGPT ---
def run_detectgpt(df, answer_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DetectGPT] Using device: {device}")

    # we’ll use GPT-2 for log-probability estimation
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    scores, preds = [], []

    for text in df[answer_name]:
        # tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            # original log probability
            outputs = model(**inputs, labels=inputs["input_ids"])
            logp = -outputs.loss.item()

        # create a perturbed version of text (simple synonym shuffle or mask)
        words = text.split()
        if len(words) > 5:
            words[len(words)//2] = "the"  # simple perturbation
        perturbed = " ".join(words)

        inputs_pert = tokenizer(perturbed, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs_pert = model(**inputs_pert, labels=inputs_pert["input_ids"])
            logp_pert = -outputs_pert.loss.item()

        curvature = logp - logp_pert
        score = np.tanh(curvature * 10)  # normalize to [-1, 1]
        pred = "AI-generated" if score > 0 else "Human-written"

        scores.append(score)
        preds.append(pred)

    df[answer_name + "_ai_detection_score"] = scores
    df[answer_name + "_ai_detection_prediction"] = preds
    return df


# --- Quick test block ---
if __name__ == "__main__":
    # Load only the first two rows to avoid long runtime
    base_dir = Path(__file__).resolve().parent.parent  # go up from modules/ to project root
    data_path = base_dir / "datasets" / "eli5_QA.parquet"

    # Load only the first 2 rows for testing
    df = pd.read_parquet(data_path).head(2)
    print("Loaded DataFrame:")
    print(df)

    # Make copies so each detector doesn’t overwrite columns
    df_bino = df.copy()
    df_roberta = df.copy()
    df_detect = df.copy()

    # Run each detector on its DataFrame
    #df_bino = run_ai_detector("binoculars", df_bino, "answer")
    #print("Binoculars done")
    df_roberta = run_ai_detector("roberta", df_roberta, "answer")
    print("Roberta done")
    df_detect = run_ai_detector("detectgpt", df_detect, "answer")
    print("DetectGPt done")

    # Combine the results into a single DataFrame view
    combined = pd.DataFrame({
        "question": df["question"],
        "answer": df["answer"],
        "roberta_score": df_roberta["answer_ai_detection_score"],
        "roberta_pred": df_roberta["answer_ai_detection_prediction"],
        "detectgpt_score": df_detect["answer_ai_detection_score"],
        "detectgpt_pred": df_detect["answer_ai_detection_prediction"],
    })

    print("\n=== Combined Detection Results ===")
    print(combined)
