import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


NUMERIC_FEATURES = [
    "frp_mw",
    "brightness_k",
    "confidence",
    "frp_norm",
    "dist_to_industry_km",
    "is_night",
    "persistence_count",
    "day_of_year",
]

CATEGORICAL_FEATURES = [
    "land_cover"
]


def build_pipeline():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(
        [
            (
                "prep",
                preprocessor,
            ),
            (
                "clf",
                model,
            ),
        ]
    )


def train(df: pd.DataFrame):

    X = df[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y = df["true_class"]

    counts = y.value_counts()

    stratify = (
        y
        if len(counts) > 1
        and counts.min() >= 2
        else None
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
    )

    pipe = build_pipeline()

    pipe.fit(
        X_train,
        y_train,
    )

    y_pred = pipe.predict(
        X_test
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    print(report)

    classes = sorted(
        y.unique()
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=classes,
    )

    joblib.dump(
        pipe,
        "fire_classifier.joblib",
    )

    return pipe, (
        X_test,
        y_test,
        y_pred,
    )


def predict_new(
    pipe,
    df_new: pd.DataFrame,
):

    X_new = df_new[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    predictions = pipe.predict(
        X_new
    )

    probabilities = (
        pipe.predict_proba(
            X_new
        )
    )

    classes = (
        pipe.named_steps[
            "clf"
        ].classes_
    )

    output = df_new.copy()

    output[
        "predicted_class"
    ] = predictions

    output[
        "predicted_confidence"
    ] = (
        probabilities.max(
            axis=1
        )
        .round(3)
    )

    for index, class_name in enumerate(
        classes
    ):
        column_name = (
            "proba_"
            + str(class_name)
            .replace(" ", "_")
            .replace("/", "")
        )

        output[
            column_name
        ] = (
            probabilities[:, index]
            .round(3)
        )

    return output
