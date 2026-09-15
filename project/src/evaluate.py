import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def confusion_pairs_table(y_true, y_pred, target_names):
    """
    Возвращает таблицу с числом ошибок для каждой пары (true, pred),
    отсортированную по убыванию — самые частые путаницы сверху.
    Это те самые "реальные числа" вместо общих слов "модель путает A и B".
    """
    cm = confusion_matrix(y_true, y_pred)
    rows = []
    n = len(target_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                rows.append({
                    "true_class": target_names[i],
                    "predicted_as": target_names[j],
                    "count": int(cm[i][j]),
                })
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df


def plot_confusion_matrix(y_true, y_pred, target_names, save_path=None, show=True):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(xticks_rotation=45, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix сохранена: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return cm


def show_errors(texts, y_true, y_pred, target_names, n=10, max_chars=300):
    """Печатает конкретные примеры ошибок модели."""
    errors = [
        (text, target_names[t], target_names[p])
        for text, t, p in zip(texts, y_true, y_pred)
        if t != p
    ]
    print(f"Всего ошибок: {len(errors)} из {len(texts)} ({len(errors)/len(texts):.1%})")
    for text, true_label, pred_label in errors[:n]:
        print(f"\nTRUE={true_label}  PRED={pred_label}")
        print(text[:max_chars].replace("\n", " "))
        print("-" * 60)
    return errors


if __name__ == "__main__":
    from baseline import run_baseline

    res = run_baseline(ngram_range=(1, 2), verbose=False)
    test = res["test"]

    plot_confusion_matrix(
        test.target, res["preds"], test.target_names,
        save_path="../results/confusion_matrix.png",
    )
    show_errors(test.data, test.target, res["preds"], test.target_names, n=10)

    pairs_df = confusion_pairs_table(test.target, res["preds"], test.target_names)
    pairs_df.to_csv("../results/confusion_pairs.csv", index=False)
    print("\nТаблица путаницы по парам классов:")
    print(pairs_df)
