"""
intensity_feedback_loop.py

AI Intensity Feedback Loop - machine learning logic only.

No Unreal Engine integration, no networking, no APIs/sockets, no UI.
Everything here is: load data -> train/evaluate a model -> save it ->
load it back -> simulate real-time physiological samples -> predict
distress -> recommend whether therapy intensity should stay the same
or be reduced.

Workflow:
    1. load_dataset(folder)      - load every .xlsx/.xls file and
                                    merge into one DataFrame.
    2. preprocess(df)            - select physiological feature
                                    columns + label, impute missing
                                    values, encode the label.
    3. train_model(X, y)         - split train/test, fit an
                                    explainable classifier
                                    (DecisionTreeClassifier or
                                    RandomForestClassifier).
    4. evaluate_model(...)        - accuracy, precision, recall,
                                    F1, confusion matrix.
    5. save_model(...) /
       load_model(...)            - persist/restore with joblib.
    6. simulate_real_time_stream  - generates synthetic "live"
                                    physiological samples drawn from
                                    the dataset's own distribution.
    7. predict_distress(...)      - runs the model on one sample,
                                    returns a distress probability.
    8. recommend_intensity(...)   - turns that probability into a
                                    "maintain" or "reduce_intensity"
                                    recommendation. This function only
                                    RECOMMENDS - it does not act on
                                    anything; wiring the recommendation
                                    to Unreal/a dashboard/a therapist
                                    UI is deliberately out of scope
                                    for this file.

Run directly to execute the full workflow end-to-end:
    python intensity_feedback_loop.py "C:\\Users\\Fatima\\Downloads\\Synthetic Data"
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("intensity_feedback_loop")


# ---------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------

@dataclass
class Config:
    data_folder: str = r"C:\Users\Fatima\Downloads\Synthetic Data"
    model_path: str = "model.pkl"

    # "decision_tree" or "random_forest" - both are explainable via
    # feature_importances_. Random forest tends to generalize better;
    # decision tree is easier to visualize/explain end-to-end.
    model_type: str = "random_forest"

    label_column: str = "ground_truth_label"

    # Physiological / psychophysiological signal columns used as
    # features. IDs, demographics, VR-task metadata, and flags are
    # deliberately excluded - this model reasons over the patient's
    # signals, not session bookkeeping.
    feature_columns: list = field(default_factory=lambda: [
        "heart_rate_bpm",
        "hrv_rmssd_ms",
        "hrv_sdnn_ms",
        "hrv_pnn50_pct",
        "hrv_lf_hf_ratio",
        "respiratory_rate_rpm",
        "respiratory_variability",
        "spo2_pct",
        "systolic_bp_mmhg",
        "diastolic_bp_mmhg",
        "gsr_us",
        "skin_conductance_level_us",
        "skin_conductance_response_us",
        "skin_temp_c",
        "eeg_alpha_uv2",
        "eeg_beta_uv2",
        "eeg_theta_uv2",
        "eeg_delta_uv2",
        "eeg_gamma_uv2",
        "frontal_alpha_asymmetry",
        "attention_index",
        "meditation_index",
        "cognitive_load",
        "mental_fatigue",
        "pupil_diameter_mm",
        "blink_rate_per_min",
        "tremor_index",
        "voice_stress_score",
        "speech_rate_wpm",
        "voice_energy_norm",
        "anxiety_score",
        "stress_score",
    ])

    test_size: float = 0.25
    random_state: int = 42

    # Distress probability at/above this triggers a "reduce intensity"
    # recommendation.
    distress_probability_threshold: float = 0.6


# ---------------------------------------------------------------------
# 1. Load + merge every Excel file in the dataset folder
# ---------------------------------------------------------------------

def load_dataset(folder: str) -> pd.DataFrame:
    """Loads every .xlsx/.xls/.csv file under `folder` and concatenates
    them into a single DataFrame."""
    paths = sorted(
        glob.glob(os.path.join(folder, "**", "*.xlsx"), recursive=True)
        + glob.glob(os.path.join(folder, "**", "*.xls"), recursive=True)
        + glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    )
    # DATA_DICTIONARY.csv describes the columns, it isn't a data file itself.
    paths = [p for p in paths if "data_dictionary" not in os.path.basename(p).lower()]
    if not paths:
        raise FileNotFoundError(f"No .xlsx/.xls/.csv files found under: {folder}")

    logger.info("Found %d file(s) to merge:", len(paths))
    frames = []
    for path in paths:
        try:
            ext = os.path.splitext(path)[1].lower()
            df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
            df["_source_file"] = os.path.basename(path)
            frames.append(df)
            logger.info("  loaded %s (%d rows)", os.path.basename(path), len(df))
        except Exception as exc:
            logger.warning("  skipped %s: %s", path, exc)

    if not frames:
        raise RuntimeError("No Excel files could be read successfully.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Merged dataset: %d total rows, %d columns", *merged.shape)
    return merged


# ---------------------------------------------------------------------
# 2. Preprocess: select columns, handle missing values, encode label
# ---------------------------------------------------------------------

@dataclass
class Preprocessed:
    X: pd.DataFrame
    y: np.ndarray
    imputer: SimpleImputer
    label_encoder: LabelEncoder
    feature_columns: list


def preprocess(df: pd.DataFrame, config: Config) -> Preprocessed:
    available_features = [c for c in config.feature_columns if c in df.columns]
    missing_features = [c for c in config.feature_columns if c not in df.columns]
    if missing_features:
        logger.info("Feature columns not present in dataset (skipped): %s", missing_features)
    if not available_features:
        raise ValueError("None of the configured feature columns were found in the dataset.")

    if config.label_column not in df.columns:
        raise ValueError(
            f"Label column '{config.label_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    working = df[available_features + [config.label_column]].copy()

    # Drop rows with a missing label - we can impute features, but we
    # can't train on an unknown target.
    working = working.dropna(subset=[config.label_column])

    # Handle missing values in the features via mean imputation.
    imputer = SimpleImputer(strategy="mean")
    X_array = imputer.fit_transform(working[available_features])
    X = pd.DataFrame(X_array, columns=available_features, index=working.index)

    # Encode the label (works whether it's already numeric, boolean,
    # or a string like "distressed"/"calm").
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(working[config.label_column])

    logger.info(
        "Preprocessed data: %d rows, %d features, classes=%s",
        len(X), X.shape[1], list(label_encoder.classes_),
    )

    return Preprocessed(
        X=X, y=y, imputer=imputer, label_encoder=label_encoder,
        feature_columns=available_features,
    )


# ---------------------------------------------------------------------
# 3. Train an explainable model (decision tree or random forest)
# ---------------------------------------------------------------------

def train_model(prep: Preprocessed, config: Config):
    X_train, X_test, y_train, y_test = train_test_split(
        prep.X, prep.y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=prep.y if len(np.unique(prep.y)) > 1 else None,
    )

    if config.model_type == "decision_tree":
        model = DecisionTreeClassifier(max_depth=6, random_state=config.random_state)
    elif config.model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=config.random_state
        )
    else:
        raise ValueError("config.model_type must be 'decision_tree' or 'random_forest'")

    logger.info("Training %s ...", config.model_type)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    log_feature_importances(model, prep.feature_columns)

    return model, X_test, y_test, metrics


# ---------------------------------------------------------------------
# 4. Evaluation: accuracy, precision, recall, F1, confusion matrix
# ---------------------------------------------------------------------

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    average = "binary" if len(np.unique(y_test)) == 2 else "macro"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    logger.info("Evaluation results:")
    logger.info("  Accuracy : %.3f", metrics["accuracy"])
    logger.info("  Precision: %.3f", metrics["precision"])
    logger.info("  Recall   : %.3f", metrics["recall"])
    logger.info("  F1-score : %.3f", metrics["f1_score"])
    logger.info("  Confusion matrix:\n%s", metrics["confusion_matrix"])

    return metrics


def log_feature_importances(model, feature_columns: list) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    importances = sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda t: t[1], reverse=True,
    )
    logger.info("Feature importances (most -> least influential):")
    for name, importance in importances[:10]:
        logger.info("  %-30s %.3f", name, importance)


# ---------------------------------------------------------------------
# 5. Save / load the trained model with joblib
# ---------------------------------------------------------------------

@dataclass
class SavedModelBundle:
    """Everything needed to run inference later: the model itself,
    plus the imputer/encoder/feature list it was trained with, so a
    freshly loaded model can preprocess new samples consistently."""
    model: object
    imputer: SimpleImputer
    label_encoder: LabelEncoder
    feature_columns: list
    distress_probability_threshold: float


def save_model(model, prep: Preprocessed, config: Config, path: Optional[str] = None) -> str:
    path = path or config.model_path
    bundle = SavedModelBundle(
        model=model,
        imputer=prep.imputer,
        label_encoder=prep.label_encoder,
        feature_columns=prep.feature_columns,
        distress_probability_threshold=config.distress_probability_threshold,
    )
    joblib.dump(bundle, path)
    logger.info("Saved trained model bundle -> %s", path)
    return path


def load_model(path: str) -> SavedModelBundle:
    bundle = joblib.load(path)
    logger.info("Loaded model bundle from %s (features: %s)", path, bundle.feature_columns)
    return bundle


# ---------------------------------------------------------------------
# 6. Simulate real-time physiological data
# ---------------------------------------------------------------------

def simulate_real_time_stream(
    reference_df: pd.DataFrame,
    feature_columns: list,
    n_samples: int = 10,
    noise_std_fraction: float = 0.05,
    random_state: Optional[int] = None,
):
    """Yields synthetic 'live' physiological samples.

    Each sample is drawn by bootstrapping a real row from the dataset
    and adding small Gaussian jitter (a fraction of that feature's
    standard deviation) to each value, so consecutive samples look
    like noisy sensor readings rather than exact repeats of history.
    This stands in for a live biosignal feed without requiring any
    actual hardware/network integration.
    """
    rng = np.random.default_rng(random_state)
    available = [c for c in feature_columns if c in reference_df.columns]
    stds = reference_df[available].std(numeric_only=True).fillna(0.0)

    for _ in range(n_samples):
        base_row = reference_df[available].sample(n=1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        noisy = {}
        for col in available:
            noise = rng.normal(0, noise_std_fraction * (stds[col] or 1.0))
            noisy[col] = float(base_row[col]) + noise if pd.notna(base_row[col]) else np.nan
        yield noisy


# ---------------------------------------------------------------------
# 7. Predict distress from one sample
# ---------------------------------------------------------------------

def predict_distress(bundle: SavedModelBundle, sample: dict) -> dict:
    """Runs the loaded model on a single sample dict and returns the
    predicted class label plus the probability of the 'distressed'
    outcome. Assumes the distress-positive class is the one whose
    encoded label sorts last after LabelEncoder (typical for binary
    labels like [0, 1] or ["calm", "distress"] -> [0, 1]); adjust
    `positive_class_index` if your label encoding differs."""
    row = pd.DataFrame([{col: sample.get(col, np.nan) for col in bundle.feature_columns}])
    row_imputed = pd.DataFrame(
        bundle.imputer.transform(row), columns=bundle.feature_columns
    )

    predicted_index = bundle.model.predict(row_imputed)[0]
    predicted_label = bundle.label_encoder.inverse_transform([predicted_index])[0]

    distress_probability = None
    if hasattr(bundle.model, "predict_proba"):
        proba = bundle.model.predict_proba(row_imputed)[0]
        positive_class_index = len(bundle.label_encoder.classes_) - 1
        distress_probability = float(proba[positive_class_index])

    return {
        "predicted_label": predicted_label,
        "distress_probability": distress_probability,
    }


# ---------------------------------------------------------------------
# 8. Recommend intensity action (recommendation only - never applies it)
# ---------------------------------------------------------------------

def recommend_intensity(prediction: dict, config: Config) -> str:
    """Turns a distress prediction into a recommendation string.
    This function does not change anything in a running session - it
    only returns a recommendation for whatever downstream
    system/therapist-facing layer chooses to act on it."""
    probability = prediction.get("distress_probability")
    if probability is None:
        # Model has no predict_proba - fall back to the predicted label.
        label = str(prediction["predicted_label"]).lower()
        return "reduce_intensity" if label in ("1", "true", "distress", "distressed") else "maintain_intensity"

    if probability >= config.distress_probability_threshold:
        return "reduce_intensity"
    return "maintain_intensity"


# ---------------------------------------------------------------------
# 9. End-to-end workflow
# ---------------------------------------------------------------------

def run_training_pipeline(config: Config):
    df = load_dataset(config.data_folder)
    prep = preprocess(df, config)
    model, X_test, y_test, metrics = train_model(prep, config)
    save_model(model, prep, config)
    return df, prep, model, metrics


def run_simulation(config: Config, reference_df: pd.DataFrame, n_samples: int = 10, delay_seconds: float = 1.0):
    bundle = load_model(config.model_path)

    logger.info("\nSimulating %d real-time samples...\n", n_samples)
    for i, sample in enumerate(
        simulate_real_time_stream(reference_df, bundle.feature_columns, n_samples=n_samples), start=1
    ):
        prediction = predict_distress(bundle, sample)
        recommendation = recommend_intensity(prediction, config)

        prob_str = (
            f"{prediction['distress_probability']:.2f}"
            if prediction["distress_probability"] is not None else "n/a"
        )
        logger.info(
            "[sample %2d] predicted=%s distress_probability=%s -> recommendation=%s",
            i, prediction["predicted_label"], prob_str, recommendation,
        )
        time.sleep(delay_seconds)


if __name__ == "__main__":
    cfg = Config()
    if len(sys.argv) > 1:
        cfg.data_folder = sys.argv[1]

    dataset, preprocessed, trained_model, eval_metrics = run_training_pipeline(cfg)
    run_simulation(cfg, reference_df=dataset, n_samples=10, delay_seconds=1.0)
