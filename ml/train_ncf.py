"""
ml/train_ncf.py

Phase 6, Stage B — Neural Collaborative Filtering

Trains a small embedding-based model on logged interactions and saves it
for use in recommender.py's hybrid ranking.

Run from the project root:
    python -m ml.train_ncf
"""

import os
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from app import create_app
from app.models import Interaction, Article

DATA_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join(DATA_DIR, "ncf_model.keras")
USER_ID_MAP_PATH = os.path.join(DATA_DIR, "ncf_user_id_map.json")
ARTICLE_ID_MAP_NCF_PATH = os.path.join(DATA_DIR, "ncf_article_id_map.json")

EMBEDDING_DIM = 16
NEGATIVES_PER_POSITIVE = 4  # how many "fake" non-interactions per real one
EPOCHS = 10
BATCH_SIZE = 64


def build_model(num_users, num_articles, embedding_dim=EMBEDDING_DIM):
    user_input = keras.Input(shape=(1,), name="user_id")
    item_input = keras.Input(shape=(1,), name="article_id")

    user_embedding = layers.Embedding(num_users, embedding_dim, name="user_embedding")(user_input)
    item_embedding = layers.Embedding(num_articles, embedding_dim, name="item_embedding")(item_input)

    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)

    x = layers.Concatenate()([user_vec, item_vec])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="interaction_score")(x)

    model = keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_training_data(app):
    with app.app_context():
        interactions = Interaction.query.all()
        all_article_ids = [a.id for a in Article.query.all()]

        if not interactions:
            raise RuntimeError("No interactions found. Log some views/likes/saves first (Phase 5).")

        # Map real DB ids -> contiguous 0..N-1 indices for the embedding layers
        user_ids = sorted({i.user_id for i in interactions})
        article_ids = sorted(all_article_ids)

        user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        article_id_map = {aid: idx for idx, aid in enumerate(article_ids)}

        # Positive pairs: every logged interaction is a positive example
        positive_pairs = {(i.user_id, i.article_id) for i in interactions}

        users_train, items_train, labels_train = [], [], []
        for (uid, aid) in positive_pairs:
            users_train.append(user_id_map[uid])
            items_train.append(article_id_map[aid])
            labels_train.append(1)

        # Negative sampling: for each user, pick random articles they
        # never interacted with, labeled 0
        user_to_positive_articles = {}
        for (uid, aid) in positive_pairs:
            user_to_positive_articles.setdefault(uid, set()).add(aid)

        for uid in user_ids:
            seen = user_to_positive_articles.get(uid, set())
            candidates = [a for a in article_ids if a not in seen]
            if not candidates:
                continue
            num_negatives = min(
                len(seen) * NEGATIVES_PER_POSITIVE, len(candidates)
            )
            sampled = random.sample(candidates, num_negatives)
            for aid in sampled:
                users_train.append(user_id_map[uid])
                items_train.append(article_id_map[aid])
                labels_train.append(0)

    return (
        np.array(users_train),
        np.array(items_train),
        np.array(labels_train),
        user_id_map,
        article_id_map,
    )


def main():
    app = create_app()
    users_train, items_train, labels_train, user_id_map, article_id_map = build_training_data(app)

    print(f"Training examples: {len(labels_train)} "
          f"({int(labels_train.sum())} positive, {len(labels_train) - int(labels_train.sum())} negative)")
    print(f"Users: {len(user_id_map)}, Articles: {len(article_id_map)}")

    model = build_model(num_users=len(user_id_map), num_articles=len(article_id_map))
    model.summary()

    model.fit(
        [users_train, items_train],
        labels_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(USER_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(user_id_map, f)
    with open(ARTICLE_ID_MAP_NCF_PATH, "w", encoding="utf-8") as f:
        json.dump(article_id_map, f)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved user/article id maps to {USER_ID_MAP_PATH}, {ARTICLE_ID_MAP_NCF_PATH}")


if __name__ == "__main__":
    main()