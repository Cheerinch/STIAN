from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def calculate_metrics(labels, preds, probs):

    acc = accuracy_score(labels, preds)

    prec = precision_score(
        labels,
        preds,
        zero_division=0
    )

    rec = recall_score(
        labels,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        preds,
        zero_division=0
    )

    auc = roc_auc_score(labels, probs)

    return acc, prec, rec, f1, auc