import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def remove_stop_words(text, lemmas, stop_words):
    return [
        lemma
        for word, lemma in zip(text.split(), lemmas.split())
        if lemma.isalpha() and word not in stop_words and lemma not in stop_words
    ]


def average_word_vectors(texts, word_vectors):
    vectors = np.zeros((len(texts), word_vectors.vector_size))
    for i, text in enumerate(texts):
        known = [word_vectors[w] for w in text.split() if w in word_vectors]
        if known:
            vectors[i] = np.mean(known, axis=0)
    return vectors


def zero_share(x):
    nonzero = x.nnz if hasattr(x, "nnz") else np.count_nonzero(x)
    return 100 * (1 - nonzero / (x.shape[0] * x.shape[1]))


def evaluate(model, x_test, y_test):
    y_true = np.asarray(y_test)
    y_pred = model.predict(x_test)
    proba = model.predict_proba(x_test)

    classes = list(model.classes_)
    true_proba = proba[np.arange(len(y_true)), [classes.index(c) for c in y_true]]

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    class_accuracy = (len(y_true) - cm.sum(axis=0) - cm.sum(axis=1) + 2 * np.diag(cm)) / len(y_true)

    report = pd.DataFrame({
        "accuracy": class_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_class_proba": [true_proba[y_true == c].mean() for c in classes],
        "support": support,
    }, index=classes)
    report.loc["macro avg"] = [*report[["accuracy", "precision", "recall", "f1", "true_class_proba"]].mean(), support.sum()]

    accuracy = accuracy_score(y_true, y_pred)
    report.loc["micro avg"] = [accuracy, accuracy, accuracy, accuracy, true_proba.mean(), support.sum()]

    mean_proba = pd.DataFrame([proba[y_true == c].mean(axis=0) for c in classes], index=classes, columns=classes)

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "report": report,
        "mean_proba": mean_proba,
    }


def plot_confusion_matrix(y_test, y_pred, labels, ax, title):
    counts = confusion_matrix(y_test, y_pred, labels=labels)
    shares = counts / counts.sum(axis=1, keepdims=True)

    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if shares[i, j] > 0.5 else "black"
            ax.text(j, i, f"{counts[i, j]}\n({shares[i, j]:.1%})", ha="center", va="center", color=colour, fontsize=9)

    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
