from binoculars import Binoculars

def run_ai_detector(detector_name, df, answer_name):
    if detector_name == "Binoculars":
        return run_binoculars(df, answer_name)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def run_binoculars(df, answer_name):
    bino = Binoculars()
    scores = []
    predictions = []

    # Iterate over the text column
    for text in df[answer_name]:
        score = bino.compute_score(text)
        prediction = bino.predict(text)
        scores.append(score)
        predictions.append(prediction)

    # Add results as new columns
    df[answer_name + "_ai_detection_score"] = scores
    df[answer_name + "_ai_detection_prediction"] = predictions

    return df
