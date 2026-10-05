"""Model construction, persistence, prediction and plain-language explanations."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from . import config
from .data import add_model_inputs
from .text import clean_text

CATEGORICAL_FEATURES = ("channel", "warranty_status", "product_family")
ARTEFACT_FORMAT = 1

# Words that carry no routing meaning; they are never shown as reasons.
_FILLER = {
    "sir", "pls", "please", "hi", "hello", "team", "thanks", "the", "for", "of", "my", "is", "to",
    "a", "good", "morning", "namaste", "urgent", "kindly", "resolve", "very", "disappointed",
    "asap", "me", "on", "in", "i", "it", "and", "with", "about", "no", "reg", "order", "this",
    "water", "air", "mixer", "induction", "room", "ceiling", "robot", "purifier", "fryer",
    "grinder", "cooktop", "heater", "fan", "vacuum", "product", "machine", "be", "can", "want",
    "help", "from", "call", "back", "dear",
}
_PAID_RE = re.compile(r"\bpaid\b|payment done")
_VAGUE_RE = re.compile(
    r"someone contact me|not happy with|complaint about|need help with|please call back regarding"
    r"|service request for|\bquery\b|\bproblem\b|issue with"
)
_FAULT_RE = re.compile(
    r"tripping|leak|gone blank|noise|burnt|not turning on|error code|stopped working|not working"
)


@dataclass(frozen=True)
class ModelSpec:
    """Hyper-parameters of one candidate model."""

    name: str
    categorical: tuple[str, ...] = ()
    classifier: str = "lr"  # "lr" | "svc" | "svc_cal" (LinearSVC + sigmoid calibration)
    C: float = 10.0
    class_weight: str | None = None
    use_char: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "categorical": list(self.categorical),
            "classifier": self.classifier,
            "C": self.C,
            "class_weight": self.class_weight,
            "use_char": self.use_char,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(
            name=d["name"],
            categorical=tuple(d["categorical"]),
            classifier=d["classifier"],
            C=float(d["C"]),
            class_weight=d["class_weight"],
            use_char=bool(d["use_char"]),
        )


# Specification chosen by the time-based model comparison (scripts/evaluate.py,
# docs/evaluation-report.md). scripts.evaluate writes the same spec to reports/selected_spec.json.
DEFAULT_SPEC = ModelSpec("svc_cal_text+product", ("product_family",), "svc_cal", 1.0)


def build_pipeline(spec: ModelSpec) -> Pipeline:
    for col in spec.categorical:
        if col not in CATEGORICAL_FEATURES:
            raise ValueError(f"{col!r} is not an allowed prediction-time feature")
    transformers: list[tuple[str, Any, Any]] = [
        (
            "word",
            TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            ),
            "text_clean",
        )
    ]
    if spec.use_char:
        transformers.append(
            (
                "char",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True),
                "text_clean",
            )
        )
    if spec.categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(spec.categorical))
        )
    features = ColumnTransformer(transformers, sparse_threshold=1.0)
    if spec.classifier == "lr":
        clf: Any = LogisticRegression(C=spec.C, max_iter=5000, class_weight=spec.class_weight, random_state=0)
    elif spec.classifier == "svc":
        clf = LinearSVC(C=spec.C, class_weight=spec.class_weight, random_state=0)
    elif spec.classifier == "svc_cal":
        # ensemble=False: one LinearSVC fitted on all rows; sigmoid calibration learnt from
        # 5-fold out-of-fold decision scores. Gives usable confidence values.
        clf = CalibratedClassifierCV(
            LinearSVC(C=spec.C, class_weight=spec.class_weight, random_state=0),
            method="sigmoid",
            cv=5,
            ensemble=False,
        )
    else:
        raise ValueError(f"unknown classifier {spec.classifier!r}")
    return Pipeline([("features", features), ("clf", clf)])


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to prediction-time inputs. Refuses frames carrying only forbidden columns."""
    if "text_clean" not in df:
        df = add_model_inputs(df)
    return df[["text_clean", *CATEGORICAL_FEATURES]]


@dataclass
class RouterModel:
    pipeline: Pipeline
    metadata: dict[str, Any] = field(default_factory=dict)
    _feature_names: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.named_steps["clf"].classes_)

    def fit(self, df: pd.DataFrame, y: pd.Series) -> RouterModel:
        self.pipeline.fit(_feature_frame(df), y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(_feature_frame(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        clf = self.pipeline.named_steps["clf"]
        x = _feature_frame(df)
        if hasattr(clf, "predict_proba"):
            return self.pipeline.predict_proba(x)
        scores = self.pipeline.decision_function(x)  # softmax fallback for margin models
        scores = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ explanations
    def route(
        self,
        request_text: str,
        channel: str | None = None,
        warranty_status: str | None = None,
        product_family: str | None = None,
    ) -> dict[str, Any]:
        """Predict one request and explain the decision in plain language."""
        row = pd.DataFrame(
            [{
                "request_text": request_text,
                "channel": channel or config.UNKNOWN,
                "warranty_status": warranty_status or config.UNKNOWN,
                "product_family": product_family or config.UNKNOWN,
            }]
        )
        row = add_model_inputs(row)
        proba = self.predict_proba(row)[0]
        order = np.argsort(proba)[::-1]
        team = self.classes[order[0]]
        confidence = float(proba[order[0]])
        alternatives = [
            {"team": self.classes[i], "probability": round(float(proba[i]), 3)} for i in order[1:3]
        ]
        reasons = self._reasons(row, team, confidence)
        return {
            "team": team,
            "confidence": round(confidence, 3),
            "confidence_band": _band(confidence),
            "reasons": reasons,
            "alternatives": alternatives,
        }

    def _contributions(self, row: pd.DataFrame, team: str) -> list[tuple[str, str, float]]:
        features = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]
        x = features.transform(_feature_frame(row)).tocsr()
        coef = np.asarray(_linear_coef(clf))
        k = self.classes.index(team)
        relative = coef[k] - coef.mean(axis=0)  # pull towards this team vs. the average team
        if self._feature_names is None:
            self._feature_names = features.get_feature_names_out()
        names = self._feature_names
        out = []
        for j, v in zip(x.indices, x.data, strict=False):
            block, _, term = names[j].partition("__")
            out.append((block, term, float(v * relative[j])))
        return out

    def _reasons(self, row: pd.DataFrame, team: str, confidence: float) -> list[str]:
        text = row["text_clean"].iloc[0]
        contribs = self._contributions(row, team)
        chosen = _key_phrases(text, contribs)

        reasons = []
        if chosen:
            quoted = ", ".join(f'"{c}"' for c in chosen)
            reasons.append(f"The request mentions {quoted}, wording that is typically routed to {team}.")

        for block, term, value in contribs:
            if block == "cat" and value > 0.5:
                col = next(c for c in CATEGORICAL_FEATURES if term.startswith(c))
                val = term[len(col) + 1:]
                label = col.replace("_", " ")
                reasons.append(f"The {label} is '{val}', which historically leans towards {team}.")

        if confidence < 0.5:
            reasons.append(
                "Low confidence: the message does not clearly describe the problem. "
                "Confirm the issue with the customer before transferring."
            )
        if _PAID_RE.search(text) and team == "Billing":
            reasons.append(
                "Note: historical routing sends requests that mention a payment to Billing. "
                "Ops policy §3 says a payment mention alone is not a billing issue; check the actual problem."
            )
        elif _VAGUE_RE.search(text):
            reasons.append(
                "Note: vague requests like this were historically queued here by default, "
                "but the team that finally resolved them varied widely."
            )
        elif team == "Filters & Consumables" and _FAULT_RE.search(text):
            reasons.append(
                "Note: historical routing sends water-purifier faults to Filters & Consumables; "
                "most of these were finally resolved by Repairs."
            )
        if not reasons:
            reasons.append(f"The overall wording is closest to past requests routed to {team}.")
        return reasons


def _key_phrases(text: str, contribs: list[tuple[str, str, float]], top: int = 3) -> list[str]:
    """Turn per-feature evidence into up to `top` short phrases copied from the request."""
    tokens = re.findall(r"\w+", text)
    if not tokens:
        return []
    word_terms: dict[str, float] = {}
    char_terms: list[tuple[str, float]] = []
    for block, term, value in contribs:
        if block == "word":
            word_terms[term] = value
        elif block == "char" and term.strip():
            char_terms.append((term, value))

    # Evidence per token position: its unigram, half of each bigram it is part of, and the
    # character n-grams found inside the word (shared across repeated occurrences).
    counts = {t: tokens.count(t) for t in set(tokens)}
    score = []
    for i, tok in enumerate(tokens):
        s = word_terms.get(tok, 0.0) / counts[tok]
        if i + 1 < len(tokens):
            s += word_terms.get(f"{tok} {tokens[i + 1]}", 0.0) / 2
        if i > 0:
            s += word_terms.get(f"{tokens[i - 1]} {tok}", 0.0) / 2
        padded = f" {tok} "
        s += sum(v for term, v in char_terms if term in padded) / counts[tok]
        score.append(0.0 if tok in _FILLER else s)

    best = max(score)
    if best <= 0:
        return []
    marked = [s >= 0.2 * best for s in score]
    phrases: list[tuple[float, int, str]] = []
    i = 0
    while i < len(tokens):
        if not marked[i]:
            i += 1
            continue
        start, end = i, i
        j = i + 1
        while j < len(tokens):
            if marked[j]:
                end = j
                j += 1
            elif j + 1 < len(tokens) and marked[j + 1] and tokens[j] in _BRIDGE:
                j += 1  # allow one connecting word ("leaking water from bottom")
            else:
                break
        phrases.append((sum(score[start:end + 1]), start, " ".join(tokens[start:end + 1])))
        i = end + 1
    phrases = sorted(phrases, reverse=True)[:top]
    return [p for _, _, p in sorted(phrases, key=lambda p: p[1])]


_BRIDGE = {"not", "of", "the", "for", "to", "from", "on", "in", "my", "is", "and", "with", "a", "under"}


def _linear_coef(clf: Any) -> np.ndarray:
    if isinstance(clf, CalibratedClassifierCV):
        return clf.calibrated_classifiers_[0].estimator.coef_
    return clf.coef_


def _band(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def save_model(model: RouterModel, path: Path = config.MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": ARTEFACT_FORMAT,
        "sklearn_version": sklearn.__version__,
        "pipeline": model.pipeline,
        "metadata": model.metadata,
    }
    joblib.dump(payload, path, compress=3)
    return path


class ModelArtefactError(RuntimeError):
    pass


def load_model(path: Path = config.MODEL_PATH) -> RouterModel:
    path = Path(path)
    if not path.exists():
        raise ModelArtefactError(
            f"Model artefact not found at {path}. Train it with `python -m scripts.train` "
            "(requires the assignment data) or copy the shared router.joblib into models/."
        )
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format") != ARTEFACT_FORMAT:
        raise ModelArtefactError(f"{path} is not a recognised router artefact")
    if payload["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"Artefact built with scikit-learn {payload['sklearn_version']}, "
            f"running {sklearn.__version__}; install requirements.txt to match.",
            stacklevel=2,
        )
    model = RouterModel(pipeline=payload["pipeline"], metadata=payload["metadata"])
    unknown = set(model.classes) - set(config.CURRENT_TEAMS)
    if unknown:
        raise ModelArtefactError(f"artefact predicts unexpected teams {sorted(unknown)}")
    return model


__all__ = [
    "CATEGORICAL_FEATURES",
    "DEFAULT_SPEC",
    "ModelArtefactError",
    "ModelSpec",
    "RouterModel",
    "build_pipeline",
    "clean_text",
    "load_model",
    "save_model",
]
