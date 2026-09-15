import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline import run_baseline


def experiment_1_ngrams():
    results = []
    for ngram_range, name in [((1, 1), "unigrams"), ((1, 2), "uni+bigrams")]:
        res = run_baseline(ngram_range=ngram_range, verbose=False)
        results.append({
            "experiment": name,
            "ngram_range": str(ngram_range),
            "vocab_size": len(res["vectorizer"].vocabulary_),
            "macro_f1": res["macro_f1"],
        })
        print(f"{name}: macro-F1={res['macro_f1']:.4f}, vocab_size={len(res['vectorizer'].vocabulary_)}")
    return pd.DataFrame(results)


def experiment_2_top_bigrams(top_n=15):
    res = run_baseline(ngram_range=(1, 2), verbose=False)
    vectorizer = res["vectorizer"]
    clf = res["model"]
    train = res["train"]

    feature_names = np.array(vectorizer.get_feature_names_out())

    report = {}
    for i, class_name in enumerate(train.target_names):
        top_idx = np.argsort(clf.coef_[i])[-100:]  # берём с запасом, потом фильтруем биграммы
        top_features = feature_names[top_idx][::-1]
        bigrams = [f for f in top_features if " " in f][:top_n]
        report[class_name] = bigrams
        print(f"\nКласс: {class_name}")
        print("Топ биграммы:", bigrams)
    return report


def experiment_4_per_class_effect():
    results = []
    per_class_f1 = {}
    for ngram_range, name in [((1, 1), "unigrams"), ((1, 2), "uni+bigrams")]:
        res = run_baseline(ngram_range=ngram_range, verbose=False)
        test = res["test"]
        f1_per_class = f1_score(test.target, res["preds"], average=None)
        per_class_f1[name] = dict(zip(test.target_names, f1_per_class))
        for class_name, f1_val in zip(test.target_names, f1_per_class):
            results.append({
                "experiment": name,
                "class": class_name,
                "f1": f1_val,
            })

    df = pd.DataFrame(results)
    pivot = df.pivot(index="class", columns="experiment", values="f1")
    pivot["delta"] = pivot["uni+bigrams"] - pivot["unigrams"]
    print(pivot)
    return pivot.reset_index()


def experiment_3_regularization():
    results = []
    for C in [0.1, 1.0, 10.0]:
        for ngram_range, name in [((1, 1), "unigrams"), ((1, 2), "uni+bigrams")]:
            res = run_baseline(ngram_range=ngram_range, C=C, verbose=False)
            results.append({
                "C": C,
                "features": name,
                "macro_f1": res["macro_f1"],
            })
            print(f"C={C}, {name}: macro-F1={res['macro_f1']:.4f}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== Эксперимент 1: униграммы vs уни+биграммы ===")
    df1 = experiment_1_ngrams()
    df1.to_csv("../results/experiment_1_ngrams.csv", index=False)

    print("\n=== Эксперимент 2: топ биграммы по классам ===")
    experiment_2_top_bigrams()

    print("\n=== Эксперимент 3: влияние регуляризации C ===")
    df3 = experiment_3_regularization()
    df3.to_csv("../results/experiment_3_regularization.csv", index=False)

    print("\n=== Эксперимент 4: влияние биграмм по классам ===")
    df4 = experiment_4_per_class_effect()
    df4.to_csv("../results/experiment_4_per_class.csv", index=False)
