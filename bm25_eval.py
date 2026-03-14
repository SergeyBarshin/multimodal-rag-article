import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_image_to_text_map(images: Iterable[dict]) -> Dict[str, str]:
    image_to_text = {}
    for item in images:
        image_id = item.get("id")
        text_id = item.get("preceding_text_id")
        if image_id and text_id:
            image_to_text[image_id] = text_id
    return image_to_text


def patch_gold_ids_with_context(
    benchmark: List[dict], image_to_text: Dict[str, str], text_id_set: Set[str]
) -> Tuple[List[dict], int]:
    patched = []
    changed = 0
    for item in benchmark:
        gold_ids = set(item.get("gold_ids", []))
        extended = set(gold_ids)
        for gid in gold_ids:
            linked_text_id = image_to_text.get(gid)
            if linked_text_id:
                extended.add(linked_text_id)

        gold_text_ids = [gid for gid in extended if gid in text_id_set]
        if len(extended) != len(gold_ids):
            changed += 1

        updated = dict(item)
        updated["gold_ids"] = list(extended)
        updated["_gold_text_ids"] = gold_text_ids
        patched.append(updated)
    return patched, changed


def build_bm25(theory: List[dict]) -> Tuple[BM25Okapi, List[str]]:
    doc_ids = [doc["id"] for doc in theory]
    tokenized_corpus = [tokenize(doc.get("text", "")) for doc in theory]
    return BM25Okapi(tokenized_corpus), doc_ids


def top_k_ids(scores: np.ndarray, doc_ids: List[str], k: int) -> List[str]:
    if k <= 0:
        return []
    k = min(k, len(doc_ids))
    idx = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in idx]


def metrics_for_query(pred_ids: List[str], gold_ids: Set[str], k: int) -> Dict[str, float]:
    top = pred_ids[:k]
    hit_positions = [rank for rank, doc_id in enumerate(top, start=1) if doc_id in gold_ids]

    recall_at_k = 1.0 if hit_positions else 0.0
    mrr_at_k = (1.0 / hit_positions[0]) if hit_positions else 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(top, start=1):
        rel = 1.0 if doc_id in gold_ids else 0.0
        if rel:
            dcg += rel / np.log2(rank + 1)

    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg_at_k = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "recall": recall_at_k,
        "mrr": mrr_at_k,
        "ndcg": ndcg_at_k,
        "first_hit_rank": float(hit_positions[0]) if hit_positions else 0.0,
    }


def evaluate(
    bm25: BM25Okapi,
    doc_ids: List[str],
    benchmark: List[dict],
    query_field: str,
    ks: List[int],
) -> Tuple[Dict[str, dict], List[dict]]:
    groups = defaultdict(list)
    per_query_rows = []
    skipped = 0
    max_k = max(ks)

    for q_idx, item in enumerate(benchmark):
        query = item.get(query_field, "")
        gold_text_ids = set(item.get("_gold_text_ids", []))
        item_type = item.get("type", "unknown")
        query_tokens = tokenize(query)

        if not gold_text_ids:
            skipped += 1
            row = {
                "query_idx": q_idx,
                "type": item_type,
                "query_field": query_field,
                "query": query,
                "query_tokens": len(query_tokens),
                "gold_text_count": 0,
                "gold_text_ids": "",
                f"pred_top{max_k}": "",
                "top1_score": "",
                "top1_minus_top5_score": "",
                "error_bucket": "no_text_gold",
            }
            for k in ks:
                row[f"recall@{k}"] = ""
                row[f"mrr@{k}"] = ""
                row[f"ndcg@{k}"] = ""
            per_query_rows.append(row)
            continue

        scores = np.asarray(bm25.get_scores(tokenize(query)))
        ranked_idx = np.argsort(scores)[::-1]
        top_idx = ranked_idx[:max_k]
        predicted = [doc_ids[i] for i in top_idx]
        predicted_scores = [float(scores[i]) for i in top_idx]

        groups["overall"].append((predicted, gold_text_ids))
        groups[item_type].append((predicted, gold_text_ids))

        top1_score = predicted_scores[0] if predicted_scores else 0.0
        top5_score = predicted_scores[min(4, len(predicted_scores) - 1)] if predicted_scores else 0.0
        score_gap = top1_score - top5_score if predicted_scores else 0.0

        row = {
            "query_idx": q_idx,
            "type": item_type,
            "query_field": query_field,
            "query": query,
            "query_tokens": len(query_tokens),
            "gold_text_count": len(gold_text_ids),
            "gold_text_ids": "|".join(sorted(gold_text_ids)),
            f"pred_top{max_k}": "|".join(predicted),
            "top1_score": round(top1_score, 6),
            "top1_minus_top5_score": round(score_gap, 6),
            "error_bucket": "",
        }

        missed_at_main_k = metrics_for_query(predicted, gold_text_ids, max_k)["recall"] == 0.0
        if missed_at_main_k:
            if len(query_tokens) <= 3:
                row["error_bucket"] = "short_query"
            elif top1_score <= 0:
                row["error_bucket"] = "no_lexical_match"
            elif len(gold_text_ids) > 1:
                row["error_bucket"] = "multi_gold_miss"
            else:
                row["error_bucket"] = "lexical_mismatch"
        else:
            row["error_bucket"] = "hit"

        for k in ks:
            m = metrics_for_query(predicted, gold_text_ids, k)
            row[f"recall@{k}"] = round(m["recall"], 6)
            row[f"mrr@{k}"] = round(m["mrr"], 6)
            row[f"ndcg@{k}"] = round(m["ndcg"], 6)
        per_query_rows.append(row)

    result = {"query_field": query_field, "total_items": len(benchmark), "skipped_items": skipped, "metrics": {}}

    for group_name, rows in groups.items():
        result["metrics"][group_name] = {"count": len(rows)}
        for k in ks:
            recall_vals = []
            mrr_vals = []
            ndcg_vals = []
            for predicted, gold_ids in rows:
                m = metrics_for_query(predicted, gold_ids, k)
                recall_vals.append(m["recall"])
                mrr_vals.append(m["mrr"])
                ndcg_vals.append(m["ndcg"])

            result["metrics"][group_name][f"recall@{k}"] = float(np.mean(recall_vals)) if rows else 0.0
            result["metrics"][group_name][f"mrr@{k}"] = float(np.mean(mrr_vals)) if rows else 0.0
            result["metrics"][group_name][f"ndcg@{k}"] = float(np.mean(ndcg_vals)) if rows else 0.0

    return result, per_query_rows


def write_metrics_csv(report: dict, ks: List[int], output_csv: Path):
    rows = []
    for group_name, vals in report["metrics"].items():
        count = vals["count"]
        for k in ks:
            rows.append(
                {
                    "query_field": report["query_field"],
                    "group": group_name,
                    "k": k,
                    "count": count,
                    "recall": vals[f"recall@{k}"],
                    "mrr": vals[f"mrr@{k}"],
                    "ndcg": vals[f"ndcg@{k}"],
                }
            )

    fieldnames = ["query_field", "group", "k", "count", "recall", "mrr", "ndcg"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(rows: List[dict], output_csv: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_error_summary(per_query_rows: List[dict], ks: List[int]) -> List[dict]:
    main_k = max(ks)
    misses = []
    for row in per_query_rows:
        val = row.get(f"recall@{main_k}")
        if val == "" or val == 1.0:
            continue
        misses.append(row)

    by_bucket = defaultdict(int)
    by_type = defaultdict(int)
    for row in misses:
        by_bucket[row["error_bucket"]] += 1
        by_type[row["type"]] += 1

    summary = []
    for bucket, cnt in sorted(by_bucket.items(), key=lambda x: x[1], reverse=True):
        summary.append({"slice": "error_bucket", "name": bucket, "miss_count": cnt})
    for typ, cnt in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        summary.append({"slice": "query_type", "name": typ, "miss_count": cnt})
    return summary


def print_report(report: dict, ks: List[int]):
    print(f"Query field: {report['query_field']}")
    print(f"Total benchmark rows: {report['total_items']}")
    print(f"Skipped (no text gold ids): {report['skipped_items']}")
    print("")

    metric_groups = report["metrics"]
    group_order = ["overall", "text", "image", "hybrid"]
    for group in group_order:
        if group not in metric_groups:
            continue
        row = metric_groups[group]
        print(f"[{group}] count={row['count']}")
        for k in ks:
            print(
                f"  Recall@{k}: {row[f'recall@{k}']:.4f} | "
                f"MRR@{k}: {row[f'mrr@{k}']:.4f} | "
                f"nDCG@{k}: {row[f'ndcg@{k}']:.4f}"
            )
        print("")


def main():
    parser = argparse.ArgumentParser(description="BM25 baseline evaluation for EE RAG benchmark.")
    parser.add_argument("--data-dir", default="data", help="Path to directory with theory.json/images.json/benchmark_final.json")
    parser.add_argument("--benchmark-file", default="benchmark_final.json", help="Benchmark filename inside data-dir")
    parser.add_argument("--query-field", default="question", choices=["question", "answer"], help="Which field to use as a query")
    parser.add_argument("--k", default="5,10", help="Comma-separated k values, e.g. 5,10")
    parser.add_argument("--output-json", default="bm25_metrics.json", help="Path to save metrics JSON")
    parser.add_argument("--output-csv", default="bm25_metrics.csv", help="Path to save summary metrics CSV")
    parser.add_argument("--per-query-csv", default="bm25_per_query.csv", help="Path to save per-query metrics CSV")
    parser.add_argument("--errors-csv", default="bm25_errors.csv", help="Path to save missed queries CSV")
    parser.add_argument("--error-summary-csv", default="bm25_error_summary.csv", help="Path to save grouped error summary CSV")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    theory_path = data_dir / "theory.json"
    images_path = data_dir / "images.json"
    benchmark_path = data_dir / args.benchmark_file

    missing = [p for p in [theory_path, images_path, benchmark_path] if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required files: {missing_str}")

    theory = load_json(theory_path)
    images = load_json(images_path)
    benchmark = load_json(benchmark_path)
    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]

    text_ids = {row["id"] for row in theory}
    image_to_text = build_image_to_text_map(images)
    benchmark, patched_count = patch_gold_ids_with_context(benchmark, image_to_text, text_ids)

    bm25, doc_ids = build_bm25(theory)
    report, per_query_rows = evaluate(bm25, doc_ids, benchmark, query_field=args.query_field, ks=ks)
    report["patched_rows"] = patched_count
    report["docs_count"] = len(doc_ids)

    print(f"Patched benchmark rows with image->text links: {patched_count}")
    print_report(report, ks)

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON: {output_path.resolve()}")

    output_csv = Path(args.output_csv)
    write_metrics_csv(report, ks, output_csv)
    print(f"Saved CSV: {output_csv.resolve()}")

    per_query_csv = Path(args.per_query_csv)
    write_csv(per_query_rows, per_query_csv)
    print(f"Saved per-query CSV: {per_query_csv.resolve()}")

    main_k = max(ks)
    misses = [r for r in per_query_rows if r.get(f"recall@{main_k}") == 0.0]
    errors_csv = Path(args.errors_csv)
    write_csv(misses, errors_csv)
    print(f"Saved errors CSV: {errors_csv.resolve()}")

    err_summary = build_error_summary(per_query_rows, ks)
    err_summary_csv = Path(args.error_summary_csv)
    write_csv(err_summary, err_summary_csv)
    print(f"Saved error summary CSV: {err_summary_csv.resolve()}")


if __name__ == "__main__":
    main()
