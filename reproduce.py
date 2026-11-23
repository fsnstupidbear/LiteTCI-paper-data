"""Recompute manuscript tables from released scores and the original expert labels."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def cell_labels(path, wanted):
    labels = {}
    for row in read_csv(path):
        paper = row["SCOPUS ID"].strip()
        if paper in wanted:
            if paper in labels:
                raise ValueError(f"Duplicate Cell source identifier: {paper}")
            labels[paper] = int(float(row["TECHNICAL_ADVANCE"]) > 0)
    require_all(labels, wanted)
    return labels


def cross_labels(path, wanted):
    labels = {}
    for row in read_csv(path):
        paper = "PMID:" + row["PMID"].strip()
        if paper in wanted:
            if not float(row["Vote"]) > 0:
                raise ValueError(f"Unexpected vote for {paper}")
            positive = row["Tag"].strip().casefold() == "technical advance"
            labels[paper] = max(labels.get(paper, 0), int(positive))
    require_all(labels, wanted)
    return labels


def require_all(labels, wanted):
    missing = wanted.difference(labels)
    if missing:
        raise ValueError(f"The upstream file is missing {len(missing)} paper IDs, including {min(missing)}")


def check_table1(manifest, labels, include_cross):
    for ref in read_csv(ROOT / "results/table1_cohorts.csv"):
        split = ref["split"]
        rows = [r for r in manifest if r["split"] == split]
        years = [int(r["publication_year"]) for r in rows]
        assert (len(rows), min(years), max(years)) == tuple(int(ref[k]) for k in ("papers", "first_year", "last_year"))
        if split == "cross_journal_test":
            assert len({r["journal_key"] for r in rows}) == 301
            if not include_cross:
                print(f"Table 1 | {split}: {len(rows)} papers; use --cross-tags to check the positive count")
                continue
        positive = sum(labels[r["paper_id"]] for r in rows)
        assert positive == int(ref["positive_papers"])
        assert f"{100 * positive / len(rows):.1f}" == ref["positive_percent"]
        print(f"Table 1 | {split}: {positive}/{len(rows)} positive")


def cohort(rows, labels, split):
    rows = [r for r in rows if r["split"] == split]
    ids = np.array([r["paper_id"] for r in rows])
    y = np.array([labels[i] for i in ids], dtype=int)
    return rows, ids, y


def scores(rows, model):
    return np.array([float(row[model]) for row in rows])


def order(ids, score):
    return np.lexsort((ids, -score))


def check_table(file, number, calculator):
    for row in read_csv(ROOT / "results" / file):
        values = calculator(row["model"])
        for column, value in values.items():
            reported = row[column]
            actual = str(value) if isinstance(value, int) else f"{value:.3f}"
            if actual != reported:
                raise ValueError(f"Table {number}, {row['model']}, {column}: {actual} != {reported}")
        print(f"Table {number} | {row['model']}: " + ", ".join(f"{k}={row[k]}" for k in values))


def check_figure2(validation):
    numeric = [int(row["figure2_order"]) for row in validation if row["figure2_order"]]
    assert sorted(numeric) == list(range(1, 513))
    points = read_csv(ROOT / "data/figure2_efficiency.csv")
    for model, expected_count in [("litetci", 10), ("qwen3_4b", 6)]:
        model_points = [r for r in points if r["model"] == model]
        assert len(model_points) == expected_count
        for r in model_points:
            throughput = int(r["papers_per_batch"]) * 1000 / float(r["mean_batch_ms"])
            np.testing.assert_allclose(throughput, float(r["papers_per_second"]))
        best = max(model_points, key=lambda r: float(r["papers_per_second"]))
        print(f"Figure 2 | {model}: {float(best['papers_per_second']):.1f} papers/s at batch {best['papers_per_batch']}")


def check_abstract_groups(path, rows):
    expected = {r["paper_id"]: r["abstract_group"] for r in rows}
    rule = json.loads((ROOT / "data/abstract_rule.json").read_text(encoding="utf-8"))
    cue = re.compile(rule["method_statement"], re.IGNORECASE)
    boundary = re.compile(rule["sentence_split"])
    checked = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["SCOPUS ID"].strip()
            if key in expected:
                found = any(cue.search(s) for s in boundary.split(row["abstract"].strip()))
                assert expected[key] == ("explicit_statement" if found else "no_match"), key
                checked.add(key)
    require_all(checked, set(expected))
    print(f"Table 4 | Abstract rule verified for {len(checked)} papers")


def paired_ap_interval(y, guided, control):
    # Exact tie handling: only the last item at each distinct score contributes to AP.
    def prep(s):
        idx = np.argsort(-s, kind="stable")
        ends = np.flatnonzero(np.r_[np.diff(s[idx]) != 0, True])
        return idx, ends

    def ap(weight, ranked):
        idx, ends = ranked
        ordered = weight[:, idx]
        n = ordered.cumsum(axis=1)[:, ends]
        positives = (ordered * y[idx]).cumsum(axis=1)[:, ends]
        precision = np.divide(positives, n, out=np.zeros(n.shape, dtype=float), where=n > 0)
        return (np.diff(positives, axis=1, prepend=0) * precision).sum(axis=1) / positives[:, -1]

    rng = np.random.default_rng(20261019)
    ranks = [prep(guided), prep(control)]
    deltas = []
    for _ in range(40):
        samples = rng.integers(0, len(y), size=(250, len(y)))
        weights = np.stack([np.bincount(sample, minlength=len(y)) for sample in samples])
        deltas.extend(ap(weights, ranks[0]) - ap(weights, ranks[1]))
    return np.quantile(deltas, [0.025 / 4, 1 - 0.025 / 4])


def reproduce(args):
    manifest = read_csv(ROOT / "data/paper_splits.csv")
    cross = {r["paper_id"] for r in manifest if r["split"] == "cross_journal_test"}
    labels = cell_labels(args.cell_tags, {r["paper_id"] for r in manifest} - cross)
    if args.cross_tags:
        labels.update(cross_labels(args.cross_tags, cross))
    check_table1(manifest, labels, bool(args.cross_tags))
    predictions = read_csv(ROOT / "data/evaluation_scores.csv")
    val, _, vy = cohort(predictions, labels, "validation")
    test, ids, y = cohort(predictions, labels, "test")
    if args.cell_papers:
        check_abstract_groups(args.cell_papers, test)
    explicit = np.array([r["abstract_group"] == "explicit_statement" for r in test])
    other = ~explicit
    assert (explicit.sum(), other.sum()) == (63, 624)

    def performance(model):
        return {"validation_ap": average_precision_score(vy, scores(val, model)),
                "test_ap": average_precision_score(y, scores(test, model))}

    def supervision(model):
        return {**performance(model), "test_auroc": roc_auc_score(y, scores(test, model))}

    def subgroups(model):
        s = scores(test, model)
        top = order(ids[other], s[other])[:125]
        return {"explicit_statement_auroc": roc_auc_score(y[explicit], s[explicit]),
                "no_match_auroc": roc_auc_score(y[other], s[other]),
                "no_match_ap": average_precision_score(y[other], s[other]),
                "no_match_hits_at_125": int(y[other][top].sum())}

    check_table("table2_performance.csv", 2, performance)
    check_table("table3_supervision.csv", 3, supervision)
    check_table("table4_abstract_subgroups.csv", 4, subgroups)
    check_figure2(val)
    guided = scores(test, "litetci")
    control = scores(test, "student_permuted_teacher")
    teacher = scores(test, "qwen3_4b")
    for model in ("litetci", "modernbert_base"):
        ranked = order(ids, scores(test, model))
        rank = np.flatnonzero(y[ranked].cumsum() >= y.sum() / 2)[0] + 1
        print(f"Half of test positives recovered | {model}: {rank} papers read")
    for model in ("litetci", "scibert"):
        print(f"Test rank correlation with teacher | {model}: {spearmanr(scores(test, model), teacher).statistic:.3f}")
    def negatives_above(s):
        return (s[y == 0][:, None] > s[y == 1]).sum(axis=0)
    print(f"Positives with fewer negatives above them than under permuted supervision: {(negatives_above(guided) < negatives_above(control)).sum()} of {y.sum()}")
    if args.bootstrap:
        lo, hi = paired_ap_interval(y, guided, control)
        print(f"Test AP gain over permuted supervision, adjusted paired interval: [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-tags", required=True, type=Path, help="Original papers_all_tags.csv from Jeon et al.")
    parser.add_argument("--cross-tags", type=Path, help="Optional original tags-1990-2000.csv from Shi and Evans")
    parser.add_argument("--cell-papers", type=Path, help="Optional original cell_all.csv from Jeon et al. to check the abstract groups")
    parser.add_argument("--bootstrap", action="store_true", help="Also recompute the 10,000-resample interval in Section 5.3")
    reproduce(parser.parse_args())
