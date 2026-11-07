"""Score every Ames tool on one set, with uncertainty, and try to beat them.

Two questions this answers that the earlier runs did not.

First, how wide are the figures already on screen. A balanced accuracy of
0.847 read off 933 structures is not an exact number; resampling the same
structures says how far it could have fallen by luck alone. Without that
interval a reader cannot tell whether 0.847 against 0.741 is a real gap or
the width of the measurement.

Second, whether putting the tools together does better than the best one
alone. Three of them return a score, so a rank average needs no fitting and
cannot be optimistic; a fitted combination can be, so it is scored only on
out-of-fold predictions. Every combination is judged against the best single
tool on the rows both answered, with a paired bootstrap on the difference,
because a combination that answers fewer rows can look better simply by
dropping the hard ones.

Run it with the model environment's interpreter, which has scikit-learn:

    external_tools/ames-env/Scripts/python.exe -I \\
      evaluation/scripts/evaluate_tool_panel.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results"
RUNS = ROOT / "evaluation" / "runs"

TRUTH = RESULTS / "ames_independent_unseen.csv"

# Where each tool's call for the independent file is stored, and whether it
# returns a score. The two in-house rows are the balanced refits, which are
# what the software ships and what the screen's figures describe.
TOOLS = {
    "rules": {"label": "전문가 규칙 (ICH M7)", "scored": False,
              "path": RESULTS / "rules_independent_predictions.csv"},
    "toxtree": {"label": "Toxtree 3.1.0", "scored": False,
                "path": RUNS / "toxtree_independent" / "toxtree_predictions.csv"},
    "qsar_toolbox": {"label": "QSAR Toolbox 4.9 OASIS", "scored": False,
                     "path": RESULTS
                     / "qsar_toolbox_independent_qsar_toolbox_oasis.csv"},
    "admet": {"label": "ADMET-AI 2.0.1", "scored": True,
              "path": RESULTS / "admet_independent_predictions.csv"},
    "epa_test": {"label": "EPA TEST 5.1.2", "scored": True,
                 "path": RESULTS / "epa_predictions.csv"},
    "ames_xgb": {"label": "자체 모델 1: fingerprint", "scored": True,
                 "path": RESULTS / "ames_balanced_independent_predictions.csv"},
    "ames_desc": {"label": "자체 모델 2: descriptor", "scored": True,
                  "path": RESULTS
                  / "ames_descriptor_balanced_independent_predictions.csv"},
}

# The score-returning tools, in the order a reader meets them on screen.
SCORED = ("ames_xgb", "ames_desc", "admet", "epa_test")
BOOTSTRAP = 2000
SEED = 20261108


def read_truth():
    ids, smiles, labels = [], [], []
    with TRUTH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] not in ("0", "1"):
                continue
            ids.append(row["id"])
            smiles.append(row["smiles"])
            labels.append(int(row["label"]))
    return ids, smiles, np.array(labels)


def read_tool(path, ids):
    """Call and score per truth id. A missing or non-binary call abstains."""
    calls, scores = {}, {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["id"]
            prediction = (row.get("prediction") or "").strip()
            calls[key] = int(prediction) if prediction in ("0", "1") else None
            raw = (row.get("score") or "").strip()
            if raw:
                try:
                    scores[key] = float(raw)
                except ValueError:
                    pass
    call = np.array([calls.get(key) if calls.get(key) is not None else -1
                     for key in ids])
    score = np.array([scores.get(key, np.nan) for key in ids], dtype=float)
    return call, score


def auc(labels, scores):
    """ROC-AUC by rank, ties averaged. None when a class is missing."""
    mask = ~np.isnan(scores)
    y, s = labels[mask], scores[mask]
    positives, negatives = (y == 1).sum(), (y == 0).sum()
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    index = 0
    while index < len(s):
        stop = index
        while stop + 1 < len(s) and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        ranks[order[index:stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def confusion(labels, calls):
    decided = calls >= 0
    y, c = labels[decided], calls[decided]
    tp = int(((y == 1) & (c == 1)).sum())
    fn = int(((y == 1) & (c == 0)).sum())
    tn = int(((y == 0) & (c == 0)).sum())
    fp = int(((y == 0) & (c == 1)).sum())
    return tp, fp, tn, fn


def balanced(labels, calls):
    """Balanced accuracy over the rows answered, or None if a class is gone."""
    tp, fp, tn, fn = confusion(labels, calls)
    if tp + fn == 0 or tn + fp == 0:
        return None
    return (tp / (tp + fn) + tn / (tn + fp)) / 2.0


def metrics(labels, calls, scores=None):
    tp, fp, tn, fn = confusion(labels, calls)
    decided = int((calls >= 0).sum())
    out = {"rows": int(len(labels)), "decided": decided,
           "coverage": decided / len(labels),
           "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "sensitivity": tp / (tp + fn) if tp + fn else None,
           "specificity": tn / (tn + fp) if tn + fp else None,
           "balanced_accuracy": balanced(labels, calls),
           "caught_of_all_positives": tp / int((labels == 1).sum())}
    if scores is not None:
        out["auc"] = auc(labels, scores)
        out["scored_rows"] = int((~np.isnan(scores)).sum())
    return out


def interval(samples, level=0.95):
    """Percentile interval of the resampled statistic."""
    clean = np.array([value for value in samples if value is not None])
    if clean.size == 0:
        return None
    low = float(np.percentile(clean, 100 * (1 - level) / 2))
    high = float(np.percentile(clean, 100 * (1 + level) / 2))
    return {"low": low, "high": high, "width": high - low,
            "resamples": int(clean.size)}


def bootstrap_rows(labels, rng, count=BOOTSTRAP):
    """Row index draws shared by every statistic, so differences stay paired."""
    size = len(labels)
    draws = []
    while len(draws) < count:
        pick = rng.integers(0, size, size)
        if (labels[pick] == 1).any() and (labels[pick] == 0).any():
            draws.append(pick)
    return draws


def rank(values):
    """Ranks in [0, 1], ties averaged, NaN kept as NaN."""
    out = np.full(len(values), np.nan)
    mask = ~np.isnan(values)
    present = values[mask]
    if present.size == 0:
        return out
    order = np.argsort(present, kind="mergesort")
    ranks = np.empty(present.size, dtype=float)
    sorted_values = present[order]
    index = 0
    while index < present.size:
        stop = index
        while (stop + 1 < present.size
               and sorted_values[stop + 1] == sorted_values[index]):
            stop += 1
        ranks[order[index:stop + 1]] = (index + stop) / 2.0
        index = stop + 1
    out[mask] = ranks / max(present.size - 1, 1)
    return out


def majority(call_stack):
    """Majority over the tools that answered; a tie abstains."""
    out = np.full(call_stack.shape[1], -1)
    for column in range(call_stack.shape[1]):
        answered = call_stack[:, column]
        answered = answered[answered >= 0]
        if answered.size == 0:
            continue
        positives = int((answered == 1).sum())
        negatives = int((answered == 0).sum())
        if positives == negatives:
            continue
        out[column] = 1 if positives > negatives else 0
    return out


def any_positive(call_stack):
    """Positive if any tool said so, negative only if every answer was negative."""
    out = np.full(call_stack.shape[1], -1)
    for column in range(call_stack.shape[1]):
        answered = call_stack[:, column]
        answered = answered[answered >= 0]
        if answered.size == 0:
            continue
        out[column] = 1 if (answered == 1).any() else 0
    return out


def stack_features(scores, calls, keys_scored, keys_called):
    """Score columns with a missing-value flag, plus alert columns as -1/0/1."""
    columns, names = [], []
    for key in keys_scored:
        column = scores[key].copy()
        missing = np.isnan(column)
        filled = column.copy()
        filled[missing] = np.nanmedian(column) if (~missing).any() else 0.0
        columns += [filled, missing.astype(float)]
        names += [f"{key}_score", f"{key}_missing"]
    for key in keys_called:
        columns.append(calls[key].astype(float))
        names.append(f"{key}_call")
    return np.column_stack(columns), names


def out_of_fold(features, labels, folds=5, seed=SEED):
    """Out-of-fold probabilities of a weighted logistic fit.

    Every row is predicted by a model that did not see it, so the figure is
    not the fit describing its own training rows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    predictions = np.full(len(labels), np.nan)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels):
        model = Pipeline([("scale", StandardScaler()),
                          ("fit", LogisticRegression(max_iter=2000,
                                                     class_weight="balanced"))])
        model.fit(features[train], labels[train])
        predictions[test] = model.predict_proba(features[test])[:, 1]
    return predictions


def best_cutoff(labels, scores):
    """The cutoff with the highest balanced accuracy, and that figure."""
    candidates = np.unique(np.round(scores[~np.isnan(scores)], 3))
    best, best_value = None, -1.0
    for cutoff in candidates:
        calls = np.where(np.isnan(scores), -1, (scores >= cutoff).astype(int))
        value = balanced(labels, calls)
        if value is not None and value > best_value:
            best, best_value = float(cutoff), value
    return best, best_value


def nested_cutoff_calls(labels, scores, folds=5, seed=SEED):
    """Calls whose cutoff was chosen on the other folds, never on the row.

    Picking the balance point on the same rows it is then scored on borrows
    from the answer. Choosing it on four fifths and applying it to the fifth
    costs nothing and removes that.
    """
    from sklearn.model_selection import StratifiedKFold

    present = ~np.isnan(scores)
    out = np.full(len(labels), -1)
    index = np.arange(len(labels))[present]
    y = labels[present]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    cutoffs = []
    for train, test in splitter.split(index.reshape(-1, 1), y):
        inner = np.full(len(labels), np.nan)
        inner[index[train]] = scores[index[train]]
        cutoff, _ = best_cutoff(labels, inner)
        cutoffs.append(cutoff)
        rows = index[test]
        out[rows] = (scores[rows] >= cutoff).astype(int)
    return out, cutoffs


def paired_auc_difference(labels, draws, score_a, score_b):
    """AUC difference on the rows both tools scored, resampled in pairs."""
    both = (~np.isnan(score_a)) & (~np.isnan(score_b))
    masked_a = np.where(both, score_a, np.nan)
    masked_b = np.where(both, score_b, np.nan)
    point = auc(labels, masked_a) - auc(labels, masked_b)
    samples = []
    for draw in draws:
        left = auc(labels[draw], masked_a[draw])
        right = auc(labels[draw], masked_b[draw])
        if left is None or right is None:
            continue
        samples.append(left - right)
    return {"shared_rows": int(both.sum()), "difference": point,
            "interval": interval(samples)}


def paired_difference(labels, draws, calls_a, calls_b):
    """Balanced-accuracy difference on the rows both answered, resampled."""
    both = (calls_a >= 0) & (calls_b >= 0)
    y = labels[both]
    a, b = calls_a[both], calls_b[both]
    point = balanced(y, a) - balanced(y, b)
    samples = []
    for draw in draws:
        pick = draw[both[draw]]
        if pick.size == 0:
            continue
        ya = labels[pick]
        if not ((ya == 1).any() and (ya == 0).any()):
            continue
        left = balanced(ya, calls_a[pick])
        right = balanced(ya, calls_b[pick])
        if left is None or right is None:
            continue
        samples.append(left - right)
    return {"shared_rows": int(both.sum()), "difference": point,
            "interval": interval(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS / "tool_panel_wide.json"))
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP)
    args = parser.parse_args()

    ids, _, labels = read_truth()
    rng = np.random.default_rng(SEED)
    draws = bootstrap_rows(labels, rng, args.resamples)
    print(f"정답 {len(labels):,}종 · 양성 {int((labels == 1).sum()):,} "
          f"· 음성 {int((labels == 0).sum()):,}")

    calls, scores = {}, {}
    for key, entry in TOOLS.items():
        call, score = read_tool(entry["path"], ids)
        calls[key] = call
        scores[key] = score

    report = {"truth_file": TRUTH.name, "rows": int(len(labels)),
              "positives": int((labels == 1).sum()),
              "negatives": int((labels == 0).sum()),
              "resamples": len(draws), "seed": SEED,
              "single": {}, "agreement": {}, "combinations": {},
              "against_best_single": {}}

    # Every tool on the one set, with the interval around each figure.
    for key, entry in TOOLS.items():
        score = scores[key] if entry["scored"] else None
        row = metrics(labels, calls[key], score)
        row["label"] = entry["label"]
        row["source"] = str(entry["path"].relative_to(ROOT)).replace("\\", "/")
        balanced_samples, auc_samples = [], []
        for draw in draws:
            balanced_samples.append(balanced(labels[draw], calls[key][draw]))
            if score is not None:
                auc_samples.append(auc(labels[draw], score[draw]))
        row["balanced_interval"] = interval(balanced_samples)
        if score is not None:
            row["auc_interval"] = interval(auc_samples)
        report["single"][key] = row
        band = row["balanced_interval"]
        print(f"{entry['label']:26s} 균형 {row['balanced_accuracy']:.3f} "
              f"[{band['low']:.3f}, {band['high']:.3f}] "
              f"답한 비율 {row['coverage']:.3f} "
              + (f"AUC {row['auc']:.3f}" if row.get("auc") else ""))

    # How much of each pair's agreement is the same tool twice over.
    keys = list(TOOLS)
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            a, b = keys[left], keys[right]
            both = (calls[a] >= 0) & (calls[b] >= 0)
            if not both.any():
                continue
            same = int((calls[a][both] == calls[b][both]).sum())
            report["agreement"][f"{a}|{b}"] = {
                "shared_rows": int(both.sum()),
                "agree": same, "fraction": same / int(both.sum())}

    # The best single tool by balanced accuracy, which every combination has
    # to beat on the rows both of them answered.
    champion = max(report["single"],
                   key=lambda key: report["single"][key]["balanced_accuracy"])
    report["best_single"] = champion
    print(f"\n단일 최고: {TOOLS[champion]['label']} "
          f"{report['single'][champion]['balanced_accuracy']:.3f}")

    candidates = {}

    # No fitting at all: the average of the score tools' ranks, cut at the
    # line that balances it. Scored on the rows every component answered.
    for name, members in (("순위 평균: 자체1+자체2+ADMET",
                           ("ames_xgb", "ames_desc", "admet")),
                          ("순위 평균: 네 도구 전부", SCORED),
                          ("순위 평균: 자체1+ADMET", ("ames_xgb", "admet"))):
        stack = np.column_stack([rank(scores[key]) for key in members])
        complete = ~np.isnan(stack).any(axis=1)
        mean = np.full(len(labels), np.nan)
        mean[complete] = stack[complete].mean(axis=1)
        cutoff, value = best_cutoff(labels, mean)
        candidates[name] = {
            "kind": "rank_mean", "members": list(members),
            "cutoff_at_balance": cutoff,
            "calls": np.where(np.isnan(mean), -1, (mean >= cutoff).astype(int)),
            "score": mean,
            "note": "적합 과정이 없다. 자른 지점은 폴드 밖에서 고른다."}

    # The same average, but over whichever of the three answered. The
    # fingerprint model abstains on 268 structures it cannot read, salts and
    # mixtures mostly, and the descriptor model answers every one of them, so
    # requiring all three to answer throws away rows for no reason.
    members = ("ames_xgb", "ames_desc", "admet")
    stack = np.column_stack([rank(scores[key]) for key in members])
    answered = (~np.isnan(stack)).sum(axis=1)
    mean = np.full(len(labels), np.nan)
    usable = answered > 0
    mean[usable] = np.nanmean(stack[usable], axis=1)
    cutoff, _ = best_cutoff(labels, mean)
    candidates["순위 평균: 답한 모델만 (자체1+자체2+ADMET)"] = {
        "kind": "rank_mean_available", "members": list(members),
        "cutoff_at_balance": cutoff,
        "calls": np.where(np.isnan(mean), -1, (mean >= cutoff).astype(int)),
        "score": mean,
        "note": "셋 중 답한 것만 평균한다. 자체 모델 1이 못 읽는 염과 혼합물을 "
                "자체 모델 2가 답하므로 답한 비율이 1.000이 된다."}

    # The plain average of the scores themselves. A rank average needs a
    # reference population to rank against, which a screen looking at four
    # materials does not have, so if the plain average does as well it is the
    # one worth having.
    for name, members in (("점수 평균: 자체1+자체2+ADMET",
                           ("ames_xgb", "ames_desc", "admet")),
                          ("점수 평균: 답한 모델만", ("ames_xgb", "ames_desc",
                                               "admet"))):
        stack = np.column_stack([scores[key] for key in members])
        mean = np.full(len(labels), np.nan)
        if name.endswith("답한 모델만"):
            usable = (~np.isnan(stack)).any(axis=1)
            mean[usable] = np.nanmean(stack[usable], axis=1)
        else:
            complete = ~np.isnan(stack).any(axis=1)
            mean[complete] = stack[complete].mean(axis=1)
        cutoff, _ = best_cutoff(labels, mean)
        candidates[name] = {
            "kind": "score_mean", "members": list(members),
            "cutoff_at_balance": cutoff,
            "calls": np.where(np.isnan(mean), -1, (mean >= cutoff).astype(int)),
            "score": mean,
            "note": "점수를 그대로 평균한다. 기준 집단이 필요 없으므로 물질 "
                    "몇 개만 보는 화면에서도 그대로 쓸 수 있다."}

    # Votes, which need no fitting either.
    for name, members, how in (
            ("다수결: 자체1+자체2+ADMET", ("ames_xgb", "ames_desc", "admet"),
             majority),
            ("다수결: 규칙+Toxtree+자체1", ("rules", "toxtree", "ames_xgb"),
             majority),
            ("하나라도 양성: 규칙+Toxtree", ("rules", "toxtree"), any_positive),
            ("하나라도 양성: 자체1+ADMET", ("ames_xgb", "admet"), any_positive)):
        stack = np.vstack([calls[key] for key in members])
        candidates[name] = {"kind": "vote", "members": list(members),
                            "calls": how(stack), "score": None}

    # A fitted combination, scored only out of fold.
    features, names = stack_features(
        scores, calls, SCORED, ("rules", "toxtree", "qsar_toolbox"))
    fitted = out_of_fold(features, labels)
    cutoff, _ = best_cutoff(labels, fitted)
    candidates["적합한 결합: 로지스틱 (폴드 밖 예측)"] = {
        "kind": "stack", "members": names, "cutoff_at_balance": cutoff,
        "calls": (fitted >= cutoff).astype(int), "score": fitted,
        "note": "5겹 교차검증의 폴드 밖 예측만 썼으므로 자기 학습 행을 "
                "되풀어 맞춘 수치가 아니다. 자른 지점은 이 자료에서 골랐다."}

    for name, entry in candidates.items():
        row = metrics(labels, entry["calls"], entry.get("score"))
        row["kind"] = entry["kind"]
        row["members"] = entry["members"]
        if entry.get("cutoff_at_balance") is not None:
            row["cutoff_at_balance"] = entry["cutoff_at_balance"]
        if entry.get("note"):
            row["note"] = entry["note"]
        samples = [balanced(labels[draw], entry["calls"][draw]) for draw in draws]
        row["balanced_interval"] = interval(samples)
        if entry.get("score") is not None:
            row["auc_interval"] = interval(
                [auc(labels[draw], entry["score"][draw]) for draw in draws])
            # The figure that owes nothing to this file: the cutoff comes
            # from the other folds, so it cannot be the cutoff that happened
            # to suit these rows.
            honest, cutoffs = nested_cutoff_calls(labels, entry["score"])
            row["nested_cutoff"] = {
                "cutoffs": cutoffs,
                "balanced_accuracy": balanced(labels, honest),
                "balanced_interval": interval(
                    [balanced(labels[draw], honest[draw]) for draw in draws]),
                **{key: value for key, value
                   in metrics(labels, honest).items()
                   if key in ("sensitivity", "specificity", "coverage",
                              "tp", "fp", "tn", "fn")}}
            entry["nested_calls"] = honest
            # And at the line the rest of the screen runs at, which owes
            # nothing to this file at all.
            at_half = np.where(np.isnan(entry["score"]), -1,
                               (entry["score"] >= 0.50).astype(int))
            row["at_0.50"] = {
                **{key: value for key, value in metrics(labels, at_half).items()
                   if key in ("sensitivity", "specificity",
                              "balanced_accuracy", "coverage")},
                "balanced_interval": interval(
                    [balanced(labels[draw], at_half[draw]) for draw in draws])}
        report["combinations"][name] = row
        compared = entry.get("nested_calls", entry["calls"])
        report["against_best_single"][name] = paired_difference(
            labels, draws, compared, calls[champion])
        if entry.get("score") is not None:
            report["against_best_single"][name]["auc"] = paired_auc_difference(
                labels, draws, entry["score"], scores[champion])
        gap = report["against_best_single"][name]
        band = gap["interval"]
        shown = row.get("nested_cutoff", row)["balanced_accuracy"]
        print(f"{name:36s} 균형 {shown:.3f} "
              f"답한 비율 {row['coverage']:.3f} · 최고 단일과의 차이 "
              f"{gap['difference']:+.3f} [{band['low']:+.3f}, {band['high']:+.3f}]"
              + (f" · AUC 차이 {gap['auc']['difference']:+.3f} "
                 f"[{gap['auc']['interval']['low']:+.3f}, "
                 f"{gap['auc']['interval']['high']:+.3f}]"
                 if gap.get("auc") else ""))

    # A combination can only be shipped if there is a set left to report it
    # on. The cross-validated figures above estimate how well the recipe
    # does, but every one of the 1,201 structures took part in choosing the
    # coefficients, so none of them is left to state a figure with. So the
    # file is cut in half by label: fit on one half, report on the other,
    # then the other way round. Both halves are reported, because a recipe
    # that only works on one of them is a recipe that found this half.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    recipes = {
        "통계 모델 셋": {"scored": ("ames_xgb", "ames_desc", "admet"),
                     "called": ()},
        "통계 모델 셋 + 규칙 둘": {"scored": ("ames_xgb", "ames_desc", "admet"),
                             "called": ("rules", "toxtree")},
        "도구 일곱 전부": {"scored": SCORED,
                      "called": ("rules", "toxtree", "qsar_toolbox")},
    }
    halves = list(StratifiedKFold(n_splits=2, shuffle=True,
                                  random_state=SEED).split(
                                      np.zeros((len(labels), 1)), labels))
    report["half_split"] = {}
    print("\n절반으로 적합하고 나머지 절반에서 재기")
    for name, recipe in recipes.items():
        features, columns = stack_features(
            scores, calls, recipe["scored"], recipe["called"])
        rounds = []
        for number, (fit_rows, report_rows) in enumerate(halves, start=1):
            model = Pipeline([("scale", StandardScaler()),
                              ("fit", LogisticRegression(
                                  max_iter=2000, class_weight="balanced"))])
            model.fit(features[fit_rows], labels[fit_rows])
            held = np.full(len(labels), np.nan)
            held[report_rows] = model.predict_proba(features[report_rows])[:, 1]
            # The cutoff is chosen on the fitting half, so the reporting half
            # decides nothing about it.
            fitted_side = np.full(len(labels), np.nan)
            fitted_side[fit_rows] = model.predict_proba(
                features[fit_rows])[:, 1]
            cutoff, _ = best_cutoff(labels, fitted_side)
            held_calls = np.where(np.isnan(held), -1,
                                  (held >= cutoff).astype(int))
            row = metrics(labels, held_calls, held)
            row["half"] = number
            row["fit_rows"] = int(len(fit_rows))
            row["cutoff_from_fitting_half"] = cutoff
            row["coefficients"] = dict(zip(
                columns, [float(value) for value
                          in model.named_steps["fit"].coef_[0]]))
            row["versus_best_single"] = paired_difference(
                labels, draws, held_calls, calls[champion])
            row["versus_best_single"]["auc"] = paired_auc_difference(
                labels, draws, held, scores[champion])
            rounds.append(row)
            gap = row["versus_best_single"]
            print(f"  {name:22s} 절반 {number} 균형 "
                  f"{row['balanced_accuracy']:.3f} 답한 비율 "
                  f"{row['coverage']:.3f} AUC {row['auc']:.3f} · 차이 "
                  f"{gap['difference']:+.3f} "
                  f"[{gap['interval']['low']:+.3f}, "
                  f"{gap['interval']['high']:+.3f}] (공유 "
                  f"{gap['shared_rows']}행)")
        report["half_split"][name] = {"rounds": rounds, "features": columns}

    # The same split for the combinations that fit nothing, so the figures
    # can be read beside the fitted ones. Only the cutoff comes from the
    # fitting half here, because there is nothing else to take from it.
    print("한쪽에서 자른 지점만 가져온 무적합 결합")
    for name in ("순위 평균: 자체1+자체2+ADMET", "점수 평균: 자체1+자체2+ADMET",
                 "점수 평균: 답한 모델만", "순위 평균: 답한 모델만 (자체1+자체2+ADMET)"):
        entry = candidates[name]
        rounds = []
        for number, (fit_rows, report_rows) in enumerate(halves, start=1):
            fitting_side = np.full(len(labels), np.nan)
            fitting_side[fit_rows] = entry["score"][fit_rows]
            cutoff, _ = best_cutoff(labels, fitting_side)
            held = np.full(len(labels), np.nan)
            held[report_rows] = entry["score"][report_rows]
            held_calls = np.where(np.isnan(held), -1,
                                  (held >= cutoff).astype(int))
            row = metrics(labels, held_calls, held)
            row["half"] = number
            row["cutoff_from_fitting_half"] = cutoff
            row["versus_best_single"] = paired_difference(
                labels, draws, held_calls, calls[champion])
            rounds.append(row)
            gap = row["versus_best_single"]
            print(f"  {name:30s} 절반 {number} 균형 "
                  f"{row['balanced_accuracy']:.3f} 답한 비율 "
                  f"{row['coverage']:.3f} AUC {row['auc']:.3f} · 차이 "
                  f"{gap['difference']:+.3f} "
                  f"[{gap['interval']['low']:+.3f}, "
                  f"{gap['interval']['high']:+.3f}] (공유 "
                  f"{gap['shared_rows']}행)")
        report["half_split"][name] = {"rounds": rounds, "features": None,
                                      "fitted": False}

    report["notes"] = [
        "모든 수치는 ames_independent_unseen.csv 한 파일에서 나왔다. 자체 "
        "모델이 적합과 선택에 쓴 행은 이 파일에 없다.",
        "구간은 같은 구조를 다시 뽑아 만든 95% 백분위 구간이다. 다른 "
        "물질에서도 같은 값이 나온다는 뜻이 아니고, 이만큼은 운으로도 "
        "움직인다는 뜻이다.",
        "결합의 '차이'는 두 쪽이 모두 답한 행만 쓴 짝지은 차이다. 답한 "
        "비율이 낮은 결합이 어려운 행을 버려 좋아 보이는 것을 막는다.",
        "자른 지점을 이 자료에서 고른 행은 그만큼 이 자료에 유리하다. "
        "AUC는 자른 지점과 무관하므로 그 영향을 받지 않는다.",
        "음성 예측은 선별 결과이며 안전성의 근거가 아니다.",
    ]

    destination = Path(args.out)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"\n기록: {destination}")


if __name__ == "__main__":
    main()
