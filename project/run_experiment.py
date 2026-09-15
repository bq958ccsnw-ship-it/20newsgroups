import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.makedirs("results", exist_ok=True)

from baseline import run_baseline
from experiments import (
    experiment_1_ngrams,
    experiment_2_top_bigrams,
    experiment_3_regularization,
    experiment_4_per_class_effect,
)
from evaluate import plot_confusion_matrix, show_errors, confusion_pairs_table
from report import generate_report


def main():
    print("=" * 70)
    print("1. BASELINE: TF-IDF (униграммы) + LinearSVC")
    print("=" * 70)
    baseline_res = run_baseline(ngram_range=(1, 1))

    print("=" * 70)
    print("2. ЭКСПЕРИМЕНТ 1: униграммы vs уни+биграммы")
    print("=" * 70)
    df1 = experiment_1_ngrams()
    df1.to_csv("results/experiment_1_ngrams.csv", index=False)
    print(df1)

    print("=" * 70)
    print("3. ЭКСПЕРИМЕНТ 2: топ биграммы по классам")
    print("=" * 70)
    top_bigrams = experiment_2_top_bigrams()

    print("=" * 70)
    print("4. ЭКСПЕРИМЕНТ 3: влияние регуляризации C")
    print("=" * 70)
    df3 = experiment_3_regularization()
    df3.to_csv("results/experiment_3_regularization.csv", index=False)
    print(df3)

    print("=" * 70)
    print("5. ЭКСПЕРИМЕНТ 4: эффект биграмм по каждому классу")
    print("=" * 70)
    df4 = experiment_4_per_class_effect()
    df4.to_csv("results/experiment_4_per_class.csv", index=False)

    print("=" * 70)
    print("6. ФИНАЛЬНАЯ МОДЕЛЬ (уни+биграммы) — confusion matrix и ошибки")
    print("=" * 70)
    final_res = run_baseline(ngram_range=(1, 2), verbose=False)
    test = final_res["test"]

    # show=False, чтобы скрипт не зависал на открытом окне графика
    plot_confusion_matrix(
        test.target, final_res["preds"], test.target_names,
        save_path="results/confusion_matrix.png",
        show=False,
    )
    errors = show_errors(test.data, test.target, final_res["preds"], test.target_names, n=10)

    pairs_df = confusion_pairs_table(test.target, final_res["preds"], test.target_names)
    pairs_df.to_csv("results/confusion_pairs.csv", index=False)
    print("\nТаблица путаницы по парам классов:")
    print(pairs_df)

    print("=" * 70)
    print("7. АВТОГЕНЕРАЦИЯ ОТЧЁТА results/RESULTS.md")
    print("=" * 70)
    generate_report(
        baseline_res=baseline_res,
        df1=df1,
        top_bigrams=top_bigrams,
        df3=df3,
        df4=df4,
        final_res=final_res,
        error_count=len(errors),
        total_count=len(test.data),
        pairs_df=pairs_df,
        save_path="results/RESULTS.md",
    )

    print("\nГотово. Результаты сохранены в папке results/ (включая RESULTS.md с реальными числами).")


if __name__ == "__main__":
    main()
