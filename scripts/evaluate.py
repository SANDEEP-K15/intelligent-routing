"""Time-based model comparison, selection, error analysis and business comparison.

Usage:  python -m scripts.evaluate

Writes
  reports/metrics.json              aggregate metrics (no row-level data)
  reports/evaluation_tables.md      tables quoted in docs/evaluation-report.md
  reports/selected_spec.json        the model specification used by scripts.train
  reports/private/validation_errors.csv   row-level errors (git-ignored)

test_unlabelled.csv is never read here.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from kestrel_router import analysis, config, metrics, rules
from kestrel_router.data import load_train, time_split
from kestrel_router.model import ModelSpec, RouterModel, build_pipeline

ALL_CAT = ("channel", "warranty_status", "product_family")

CANDIDATES: list[ModelSpec] = [
    ModelSpec("lr_word_text", (), "lr", 10.0, None, use_char=False),
    ModelSpec("lr_text", (), "lr", 10.0),
    ModelSpec("lr_text_balanced", (), "lr", 10.0, "balanced"),
    ModelSpec("lr_text+channel+warranty", ("channel", "warranty_status"), "lr", 10.0),
    ModelSpec("lr_text+product", ("product_family",), "lr", 10.0),
    ModelSpec("lr_text+all", ALL_CAT, "lr", 10.0),
    ModelSpec("lr_text+product_balanced", ("product_family",), "lr", 10.0, "balanced"),
    ModelSpec("lr_text+product_C1", ("product_family",), "lr", 1.0),
    ModelSpec("lr_text+product_C3", ("product_family",), "lr", 3.0),
    ModelSpec("lr_text+product_C30", ("product_family",), "lr", 30.0),
    ModelSpec("svc_text", (), "svc", 1.0),
    ModelSpec("svc_text+product", ("product_family",), "svc", 1.0),
    ModelSpec("svc_text+all", ALL_CAT, "svc", 1.0),
    ModelSpec("svc_cal_text+product", ("product_family",), "svc_cal", 1.0),
    ModelSpec("svc_cal_text+all", ALL_CAT, "svc_cal", 1.0),
]
# Raw LinearSVC has no probability output; the API must return a confidence, so it is
# reported for comparison but not eligible for selection.
SELECTABLE_CLASSIFIERS = ("lr", "svc_cal")
# Simplest model within this many accuracy points of the best mean is selected.
SELECTION_TOLERANCE = 0.003


def complexity(spec: ModelSpec) -> tuple:
    return (len(spec.categorical), spec.classifier != "lr", spec.use_char, spec.class_weight is not None)


def fit_predict(spec: ModelSpec, tr: pd.DataFrame, va: pd.DataFrame, target: str = "team") -> np.ndarray:
    model = RouterModel(build_pipeline(spec)).fit(tr, tr[target])
    return model.predict(va)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in zip(df.index, df.itertuples(index=False), strict=True):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def error_breakdown(va: pd.DataFrame, pred: np.ndarray, by: str) -> pd.DataFrame:
    d = va.assign(correct=(pred == va["team"].to_numpy()))
    g = d.groupby(by)["correct"].agg(["size", "mean"])
    g.columns = ["rows", "accuracy"]
    g["errors"] = (g["rows"] * (1 - g["accuracy"])).round().astype(int)
    g["share_of_errors"] = g["errors"] / max(int((~d["correct"]).sum()), 1)
    return g.sort_values("errors", ascending=False)


def main() -> None:
    df = analysis.tag(load_train())
    config.PRIVATE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"folds": {}, "candidates": {}}
    tables: list[str] = []

    folds = {name: time_split(df, name) for name in config.FOLDS}
    for name, (tr, va) in folds.items():
        out["folds"][name] = {
            "train_rows": len(tr), "val_rows": len(va),
            "train_period": [str(tr.created_at.min()), str(tr.created_at.max())],
            "val_period": [str(va.created_at.min()), str(va.created_at.max())],
        }

    # ---------------------------------------------------------------- baselines + candidates
    rows = []
    for name, (tr, va) in folds.items():
        majority = tr["team"].mode()[0]
        r = metrics.score(va["team"], [majority] * len(va))
        rows.append(("majority (" + majority + ")", name, r))
        rows.append(("rules/keywords", name, metrics.score(va["team"], rules.predict(va))))
    for spec in CANDIDATES:
        for name, (tr, va) in folds.items():
            t0 = time.perf_counter()
            pred = fit_predict(spec, tr, va)
            res = metrics.score(va["team"], pred)
            res["fit_predict_seconds"] = round(time.perf_counter() - t0, 2)
            rows.append((spec.name, name, res))
            print(f"{spec.name:28s} {name:9s} acc={res['accuracy']:.4f} f1={res['macro_f1']:.4f}")

    comp = pd.DataFrame(
        [{"model": m, "fold": f, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]} for m, f, r in rows]
    )
    wide = comp.pivot_table(index="model", columns="fold", values=["accuracy", "macro_f1"], sort=False)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide["accuracy_mean"] = wide[["accuracy_primary", "accuracy_backtest"]].mean(axis=1)
    wide = wide.sort_values("accuracy_mean", ascending=False)
    out["candidates"] = wide.round(4).to_dict(orient="index")

    # ---------------------------------------------------------------- selection
    selectable = [s for s in CANDIDATES if s.classifier in SELECTABLE_CLASSIFIERS]
    ml = wide.loc[[s.name for s in selectable]]
    best = ml["accuracy_mean"].max()
    eligible = [s for s in selectable if ml.loc[s.name, "accuracy_mean"] >= best - SELECTION_TOLERANCE]
    selected = min(eligible, key=complexity)
    out["selection"] = {
        "rule": (f"simplest candidate (fewest input fields, then logistic regression before calibrated SVC) "
                 f"within {SELECTION_TOLERANCE:.1%} of the best mean accuracy over both folds; "
                 f"only models that output a confidence are eligible"),
        "best_mean_accuracy": round(float(best), 4),
        "selected": selected.as_dict(),
        "selected_mean_accuracy": round(float(ml.loc[selected.name, "accuracy_mean"]), 4),
    }
    (config.REPORTS_DIR / "selected_spec.json").write_text(json.dumps(selected.as_dict(), indent=2))
    print("SELECTED", selected)

    tables.append("## Model comparison (accuracy / macro-F1 against normalised team_label)\n")
    tables.append(md_table(wide[["accuracy_primary", "accuracy_backtest", "accuracy_mean",
                                 "macro_f1_primary", "macro_f1_backtest"]]))

    # ---------------------------------------------------------------- selected model, both folds
    sel: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for name, (tr, va) in folds.items():
        pred = fit_predict(selected, tr, va)
        preds[name] = pred
        res = metrics.score(va["team"], pred)
        monthly = (
            va.assign(ok=pred == va["team"].to_numpy())
            .groupby(va.created_at.dt.to_period("M"))["ok"].mean().round(4)
        )
        res["monthly_accuracy"] = {str(k): float(v) for k, v in monthly.items()}
        rng = np.random.default_rng(7)
        ok = (pred == va["team"].to_numpy()).astype(float)
        boots = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(2000)]
        res["accuracy_95ci"] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        sel[name] = res
    out["selected_model"] = sel

    tr, va = folds["primary"]
    pred = preds["primary"]
    pt = metrics.per_team(va["team"], pred)
    cm = metrics.confusion(va["team"], pred)
    out["selected_model"]["primary"]["per_team"] = pt.round(4).to_dict(orient="index")
    out["selected_model"]["primary"]["confusion_matrix"] = cm.to_dict(orient="index")
    tables.append("\n## Selected model — per-team metrics (primary fold, Apr–Jun 2026)\n")
    tables.append(md_table(pt))
    tables.append("\n## Selected model — confusion matrix (primary fold; rows = true team_label)\n")
    tables.append(md_table(cm, "{}"))
    tb, vb = folds["backtest"]
    tables.append("\n## Selected model — per-team metrics (backtest fold, Jan–Mar 2026)\n")
    tables.append(md_table(metrics.per_team(vb["team"], preds["backtest"])))

    # ---------------------------------------------------------------- error analysis (primary fold)
    tables.append("\n## Error analysis (primary fold)\n")
    ea = {}
    for by, label in [("team", "true team_label"), ("pattern", "request pattern"),
                      ("multi_intent", "multiple intents"), ("product_family", "product family"),
                      ("channel", "channel"), ("warranty_status", "warranty status")]:
        b = error_breakdown(va, pred, by)
        ea[by] = b.round(4).to_dict(orient="index")
        tables.append(f"\n### Errors by {label}\n")
        tables.append(md_table(b))
    out["error_analysis"] = ea

    err = va.assign(pred=pred)[pred != va["team"].to_numpy()]
    top = (
        err.groupby(["template", "team", "pred"]).size().rename("errors").reset_index()
        .sort_values("errors", ascending=False).head(15).set_index("template")
    )
    tables.append("\n### Most frequent error patterns (templated wording; product names shown as <p>)\n")
    tables.append(md_table(top, "{}"))
    err[["request_id", "created_at_ist", "channel", "product_family", "warranty_status",
         "request_text", "text_clean", "pattern", "team", "pred", "final_team_norm"]].to_csv(
        config.PRIVATE_REPORTS_DIR / "validation_errors.csv", index=False)

    # Label consistency: how often does the same template get different labels in training?
    purity = tr.groupby("template")["team"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    va_purity = va["template"].map(purity)
    out["label_noise"] = {
        "val_rows_with_template_seen_in_train": round(float(va_purity.notna().mean()), 4),
        "train_label_purity_weighted": round(float(
            tr.groupby("template")["team"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(tr)), 4),
    }

    # ---------------------------------------------------------------- A vs B: contract vs outcome
    mis_train = df[df["team"] != df["final_team_norm"]]
    avg_transfers = float(mis_train["transfers"].mean())
    business: dict[str, Any] = {
        "avg_transfers_per_misrouted_request_all_train": round(avg_transfers, 3),
        "bot_label_vs_final_team_all_train": round(float((df["team"] == df["final_team_norm"]).mean()), 4),
    }
    for name, (trf, vaf) in folds.items():
        p = preds[name]
        final = vaf["final_team_norm"].to_numpy()
        bot_ok = vaf["team"].to_numpy() == final
        model_ok = p == final
        months = vaf.created_at.dt.to_period("M").nunique()
        # Analysis-only model trained on final_team (NOT used for predictions.csv).
        p_final = fit_predict(selected, trf, vaf, target="final_team_norm")
        alt_ok = p_final == final
        business[name] = {
            "A_model_vs_team_label_accuracy": round(float((p == vaf["team"].to_numpy()).mean()), 4),
            "B_bot_label_vs_final_team": round(float(bot_ok.mean()), 4),
            "B_model_vs_final_team": round(float(model_ok.mean()), 4),
            "B_analysis_only_final_team_model_vs_final_team": round(float(alt_ok.mean()), 4),
            "analysis_only_final_team_model_vs_team_label": round(float((p_final == vaf["team"].to_numpy()).mean()), 4),
            "misrouted_per_month": {
                "bot_labels": round(float((~bot_ok).sum() / months), 1),
                "team_label_model": round(float((~model_ok).sum() / months), 1),
                "final_team_model_analysis_only": round(float((~alt_ok).sum() / months), 1),
            },
            "estimated_misroute_cost_per_month_rs": {
                key: round(metrics.misroute_cost(int((~ok).sum()), avg_transfers) / months)
                for key, ok in (("bot_labels", bot_ok), ("team_label_model", model_ok),
                                ("final_team_model_analysis_only", alt_ok))
            },
            "requests_per_month": round(len(vaf) / months, 1),
        }
        pattern_b = vaf.assign(bot_ok=bot_ok, model_ok=model_ok, alt_ok=alt_ok).groupby("pattern")[
            ["bot_ok", "model_ok", "alt_ok"]].mean().round(4)
        business[name]["final_team_agreement_by_pattern"] = pattern_b.to_dict(orient="index")
        if name == "primary":
            tables.append("\n## Agreement with final_team by request pattern (primary fold)\n")
            tables.append(md_table(pattern_b.rename(columns={
                "bot_ok": "bot label = final", "model_ok": "our model = final",
                "alt_ok": "final_team model (analysis only) = final"})))
    out["business"] = business
    out["cost"] = metrics.monthly_model_cost(700)

    # ---------------------------------------------------------------- latency (selected spec, primary fit)
    model = RouterModel(build_pipeline(selected)).fit(tr, tr["team"])
    sample = va.head(300)
    t0 = time.perf_counter()
    for r in sample.itertuples():
        model.route(r.request_text, r.channel, r.warranty_status, r.product_family)
    per_route_ms = (time.perf_counter() - t0) / len(sample) * 1000
    t0 = time.perf_counter()
    model.predict(va)
    batch_ms = (time.perf_counter() - t0) / len(va) * 1000
    out["latency_ms"] = {"single_request_with_explanation": round(per_route_ms, 2),
                         "batch_per_request": round(batch_ms, 4)}

    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    (config.REPORTS_DIR / "evaluation_tables.md").write_text(
        "# Evaluation tables (generated by scripts/evaluate.py)\n\n" + "\n".join(tables) + "\n", encoding="utf-8")
    summary = {k: out[k] for k in ("selection", "business", "latency_ms", "label_noise")}
    print(json.dumps(summary, indent=2, default=str))
    for name in folds:
        print(name, {k: v for k, v in sel[name].items() if k not in ("per_team", "confusion_matrix")})


if __name__ == "__main__":
    main()
