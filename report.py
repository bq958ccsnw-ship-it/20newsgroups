import datetime


def generate_report(
    baseline_res,
    df1,
    top_bigrams,
    df3,
    df4,
    final_res,
    error_count,
    total_count,
    pairs_df,
    save_path="../results/RESULTS.md",
):
    lines = []
    lines.append("# Результаты экспериментов\n")
    lines.append(f"_Автоматически сгенерировано: {datetime.datetime.now():%Y-%m-%d %H:%M}_\n")

    # --- Baseline ---
    lines.append("## Baseline\n")
    lines.append(
        f"TF-IDF (униграммы) + LinearSVC. "
        f"Macro-F1 = **{baseline_res['macro_f1']:.4f}**, "
        f"размер словаря = {len(baseline_res['vectorizer'].vocabulary_)}.\n"
    )

    # --- Эксперимент 1 ---
    lines.append("## Эксперимент 1: униграммы vs уни+биграммы\n")
    lines.append("| Признаки | Размер словаря | Macro-F1 |")
    lines.append("|---|---|---|")
    for _, row in df1.iterrows():
        lines.append(f"| {row['experiment']} | {row['vocab_size']} | {row['macro_f1']:.4f} |")
    delta = df1.iloc[1]["macro_f1"] - df1.iloc[0]["macro_f1"]
    direction = "улучшили" if delta > 0 else "ухудшили" if delta < 0 else "не изменили"
    lines.append(
        f"\nБиграммы {direction} результат на {delta:+.4f} macro-F1 "
        f"при росте словаря в {df1.iloc[1]['vocab_size'] / df1.iloc[0]['vocab_size']:.1f} раз.\n"
    )

    # --- Эксперимент 2: топ биграммы ---
    lines.append("## Эксперимент 2: топ-биграммы по классам\n")
    for class_name, bigrams in top_bigrams.items():
        lines.append(f"**{class_name}**: {', '.join(bigrams[:10]) if bigrams else '(биграмм в топе не найдено)'}\n")

    # --- Эксперимент 3: регуляризация ---
    lines.append("## Эксперимент 3: влияние регуляризации (C)\n")
    lines.append("| C | Признаки | Macro-F1 |")
    lines.append("|---|---|---|")
    for _, row in df3.iterrows():
        lines.append(f"| {row['C']} | {row['features']} | {row['macro_f1']:.4f} |")
    lines.append("")

    # --- Эксперимент 4: эффект по классам ---
    lines.append("## Эксперимент 4: эффект биграмм по каждому классу\n")
    lines.append("| Класс | F1 (униграммы) | F1 (уни+биграммы) | Δ |")
    lines.append("|---|---|---|---|")
    for _, row in df4.iterrows():
        lines.append(
            f"| {row['class']} | {row['unigrams']:.4f} | {row['uni+bigrams']:.4f} | {row['delta']:+.4f} |"
        )
    best_class = df4.loc[df4["delta"].idxmax(), "class"]
    lines.append(f"\nСильнее всего биграммы помогли классу **{best_class}**.\n")

    # --- Финальная модель и ошибки ---
    lines.append("## Финальная модель: ошибки\n")
    lines.append(f"Macro-F1 = **{final_res['macro_f1']:.4f}**. ")
    lines.append(f"Всего ошибок: {error_count} из {total_count} ({error_count/total_count:.1%}).\n")

    lines.append("### Самые частые путаницы (пары классов)\n")
    lines.append("| Истинный класс | Предсказан как | Количество |")
    lines.append("|---|---|---|")
    for _, row in pairs_df.head(10).iterrows():
        lines.append(f"| {row['true_class']} | {row['predicted_as']} | {row['count']} |")

    content = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nОтчёт сохранён: {save_path}")
    return content
