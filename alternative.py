from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

from data import load_data


def run_alternative(model_name="all-MiniLM-L6-v2", verbose=True):
    from sentence_transformers import SentenceTransformer

    train, test = load_data()

    model = SentenceTransformer(model_name)
    X_train = model.encode(train.data, show_progress_bar=verbose)
    X_test = model.encode(test.data, show_progress_bar=verbose)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train.target)
    preds = clf.predict(X_test)

    macro_f1 = f1_score(test.target, preds, average="macro")

    if verbose:
        print(f"Модель эмбеддингов: {model_name}")
        print(f"Macro-F1: {macro_f1:.4f}\n")
        print(classification_report(test.target, preds, target_names=train.target_names))

    return {
        "preds": preds,
        "macro_f1": macro_f1,
        "train": train,
        "test": test,
    }


if __name__ == "__main__":
    run_alternative()
