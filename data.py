from sklearn.datasets import fetch_20newsgroups

CATEGORIES = ["rec.autos", "rec.motorcycles", "misc.forsale"]


def load_data(remove_metadata=True):
    """
    Загружает train/test выборки 20 Newsgroups для выбранных категорий.

    remove_metadata=True убирает headers/footers/quotes, чтобы модель
    не "читерила" по email-подписям и цитатам, а училась на содержании текста.
    """
    remove = ("headers", "footers", "quotes") if remove_metadata else ()

    train = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=remove,
        shuffle=True,
        random_state=42,
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        remove=remove,
        shuffle=True,
        random_state=42,
    )
    return train, test


if __name__ == "__main__":
    train, test = load_data()
    print("Классы:", train.target_names)
    print("Размер train:", len(train.data))
    print("Размер test:", len(test.data))

    from collections import Counter

    print("Баланс классов (train):", Counter(train.target))
    print("Баланс классов (test):", Counter(test.target))

    print("\nПример текста из train:\n", train.data[0][:300])
