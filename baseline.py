from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

from data import load_data


def run_baseline(ngram_range=(1, 1), C=1.0, verbose=True):
    train, test = load_data()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_df=0.9,
        min_df=2,
        ngram_range=ngram_range,
    )

    X_train = vectorizer.fit_transform(train.data)
    X_test = vectorizer.transform(test.data)

    clf = LinearSVC(C=C)
    clf.fit(X_train, train.target)
    preds = clf.predict(X_test)

    macro_f1 = f1_score(test.target, preds, average="macro")

    if verbose:
        print(f"ngram_range={ngram_range}, C={C}")
        print(f"Vocab size: {len(vectorizer.vocabulary_)}")
        print(f"Macro-F1: {macro_f1:.4f}\n")
        print(classification_report(test.target, preds, target_names=train.target_names))

    return {
        "vectorizer": vectorizer,
        "model": clf,
        "preds": preds,
        "macro_f1": macro_f1,
        "train": train,
        "test": test,
    }


if __name__ == "__main__":
    run_baseline()
