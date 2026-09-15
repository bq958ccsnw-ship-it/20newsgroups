# Результаты экспериментов

_Автоматически сгенерировано: 2026-09-13 12:09_

## Baseline

TF-IDF (униграммы) + LinearSVC. Macro-F1 = **0.8510**, размер словаря = 8121.

## Эксперимент 1: униграммы vs уни+биграммы

| Признаки | Размер словаря | Macro-F1 |
|---|---|---|
| unigrams | 8121 | 0.8510 |
| uni+bigrams | 16330 | 0.8592 |

Биграммы улучшили результат на +0.0082 macro-F1 при росте словаря в 2.0 раз.

## Эксперимент 2: топ-биграммы по классам

**misc.forsale**: make offer, hell ios, brand new, junk mail

**rec.autos**: wanted know, good luck, master cylinder, don understand, stuff deleted, rec autos

**rec.motorcycles**: edu breath, citizens arrest, sex life

## Эксперимент 3: влияние регуляризации (C)

| C | Признаки | Macro-F1 |
|---|---|---|
| 0.1 | unigrams | 0.8608 |
| 0.1 | uni+bigrams | 0.8607 |
| 1.0 | unigrams | 0.8510 |
| 1.0 | uni+bigrams | 0.8592 |
| 10.0 | unigrams | 0.8361 |
| 10.0 | uni+bigrams | 0.8493 |

## Эксперимент 4: эффект биграмм по каждому классу

| Класс | F1 (униграммы) | F1 (уни+биграммы) | Δ |
|---|---|---|---|
| misc.forsale | 0.9050 | 0.9105 | +0.0055 |
| rec.autos | 0.8183 | 0.8286 | +0.0103 |
| rec.motorcycles | 0.8296 | 0.8385 | +0.0090 |

Сильнее всего биграммы помогли классу **rec.autos**.

## Финальная модель: ошибки

Macro-F1 = **0.8592**. 
Всего ошибок: 168 из 1184 (14.2%).

### Самые частые путаницы (пары классов)

| Истинный класс | Предсказан как | Количество |
|---|---|---|
| rec.motorcycles | rec.autos | 67 |
| rec.autos | rec.motorcycles | 33 |
| misc.forsale | rec.autos | 29 |
| misc.forsale | rec.motorcycles | 15 |
| rec.autos | misc.forsale | 15 |
| rec.motorcycles | misc.forsale | 9 |