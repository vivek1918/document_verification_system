#!/usr/bin/env python3
"""
Evaluation script for document verification pipeline.

Fixed & improved:
- Robust JSON loading (encoding fallback).
- Rule-name mapping between evaluator and verifier outputs.
- Populates details for debugging (missing preds, person mismatches, rule mismatches).
- Default output saved under logs/metrics/.
- Uses UTF-8 for reading/writing.
"""

import argparse
import json
from typing import Dict, Any, List
from pathlib import Path
import sys

# Map short rule keys used by the evaluator to the actual keys produced by verifier
RULE_KEY_MAP = {
    "rule_1": "rule_1_name_match",
    "rule_2": "rule_2_dob_match",
    "rule_3": "rule_3_address_match",
    "rule_4": "rule_4_phone_match",
    "rule_5": "rule_5_father_name_match",
    "rule_6": "rule_6_pan_format",
    "rule_7": "rule_7_aadhaar_format"
}


def _load_json_file_with_fallback(path: str):
    """
    Try to load a JSON file using multiple encodings to avoid common Windows errors.
    Raises a RuntimeError with context if all attempts fail.
    """
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_exc = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Failed to read JSON file {path} with encodings {encodings}. Last error: {last_exc}")


def _normalize_status_label(s: str) -> str:
    """Normalize different labels to canonical 'PASS' or 'FAIL' (case-insensitive)."""
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("pass", "passed", "verified", "true", "ok", "success"):
        return "PASS"
    if s in ("fail", "failed", "rejected", "false", "no", "invalid"):
        return "FAIL"
    return s.upper()  # fallback: return as-is uppercased



# Build reverse map so we can find either key style
REVERSE_RULE_KEY_MAP = {v: k for k, v in RULE_KEY_MAP.items()}

def _get_rules_container(obj: dict):
    """Return the dict that contains per-rule results within a person object.
    Tries several common keys.
    """
    if not isinstance(obj, dict):
        return {}
    for k in ("verification_results", "verification", "verification_result", "rules", "checks", "verificationResults", "verification_data"):
        if k in obj and isinstance(obj[k], dict):
            return obj[k]
    # sometimes verification is nested under a 'result' or 'meta' section:
    for k in ("result", "meta", "data", "verification_summary"):
        sub = obj.get(k)
        if isinstance(sub, dict):
            for candidate in ("verification_results", "verification", "rules", "checks"):
                if candidate in sub and isinstance(sub[candidate], dict):
                    return sub[candidate]
    return {}


def _find_rule_key_pair(gt_rules: Dict[str, Any], pred_rules: Dict[str, Any], short_rule: str):
    """
    For a given short_rule (like 'rule_1'), find the actual keys present in gt_rules and pred_rules.
    Returns tuple (gt_key, pred_key) where either may be None if missing.
    Preference order: short key, long key.
    """
    short = short_rule
    longk = RULE_KEY_MAP.get(short_rule)
    gt_key = None
    pred_key = None

    # Find gt key
    if short in gt_rules:
        gt_key = short
    elif longk in gt_rules:
        gt_key = longk
    else:
        # maybe gt used other naming; check reverse map
        for k in gt_rules:
            if REVERSE_RULE_KEY_MAP.get(k) == short:
                gt_key = k
                break

    # Find pred key (same logic)
    if short in pred_rules:
        pred_key = short
    elif longk in pred_rules:
        pred_key = longk
    else:
        for k in pred_rules:
            if REVERSE_RULE_KEY_MAP.get(k) == short:
                pred_key = k
                break

    return gt_key, pred_key

def evaluate_predictions(ground_truth: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
    """Evaluate prediction accuracy against ground truth (robust to label/key variations)."""

    gt_dict = {gt['person_id']: gt for gt in ground_truth}
    pred_dict = {pred['person_id']: pred for pred in predictions}

    results = {
        'overall_accuracy': 0.0,
        'rule_accuracy': {},
        'person_level_accuracy': 0.0,
        'counts': {
            "gt_count": len(gt_dict),
            "pred_count": len(pred_dict),
            "matched_persons": 0,
            "missing_predictions": 0
        },
        'details': []
    }

    # initialize rule counters (short keys)
    short_rules = list(RULE_KEY_MAP.keys())
    for r in short_rules:
        results['rule_accuracy'][r] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

    total_rules = 0
    correct_rules = 0
    correct_persons = 0

    for person_id, gt_data in gt_dict.items():
        if person_id not in pred_dict:
            results['counts']['missing_predictions'] += 1
            results['details'].append({
                "person_id": person_id,
                "type": "missing_prediction",
                "message": "No prediction found for this ground-truth person"
            })
            continue

        results['counts']['matched_persons'] += 1
        pred_data = pred_dict[person_id]

        # Normalize overall-status labels before comparing
        gt_status_raw = gt_data.get('overall_status')
        pred_status_raw = pred_data.get('overall_status')
        gt_status = _normalize_status_label(gt_status_raw)
        pred_status = _normalize_status_label(pred_status_raw)

        if gt_status == pred_status:
            correct_persons += 1
        else:
            results['details'].append({
                "person_id": person_id,
                "type": "overall_status_mismatch",
                "gt_raw": gt_status_raw,
                "pred_raw": pred_status_raw,
                "gt_normalized": gt_status,
                "pred_normalized": pred_status
            })

        # Rule-level comparison
        gt_rules = _get_rules_container(gt_data)
        pred_rules = _get_rules_container(pred_data)


        person_record = {"person_id": person_id, "rule_mismatches": []}

        for short_rule in short_rules:
            gt_key, pred_key = _find_rule_key_pair(gt_rules, pred_rules, short_rule)

            if not gt_key and not pred_key:
                # nothing to compare for this rule for this person
                continue

            # Both sides present?
            if gt_key and pred_key:
                gt_r_raw = gt_rules.get(gt_key, {}).get('status')
                pred_r_raw = pred_rules.get(pred_key, {}).get('status')
                gt_r = _normalize_status_label(gt_r_raw)
                pred_r = _normalize_status_label(pred_r_raw)

                results['rule_accuracy'][short_rule]['total'] += 1
                total_rules += 1

                if gt_r == pred_r:
                    results['rule_accuracy'][short_rule]['correct'] += 1
                    correct_rules += 1
                else:
                    person_record['rule_mismatches'].append({
                        "rule": short_rule,
                        "gt_key": gt_key,
                        "pred_key": pred_key,
                        "gt_raw": gt_r_raw,
                        "pred_raw": pred_r_raw,
                        "gt_normalized": gt_r,
                        "pred_normalized": pred_r
                    })
            else:
                # one side missing: treat as mismatch for diagnostics but do not increment total rule checks
                person_record['rule_mismatches'].append({
                    "rule": short_rule,
                    "gt_key": gt_key,
                    "pred_key": pred_key,
                    "note": "rule missing on one side"
                })

        if person_record['rule_mismatches']:
            results['details'].append(person_record)

    # finalize counts
    gt_total = len(gt_dict)
    if gt_total > 0:
        results['person_level_accuracy'] = correct_persons / gt_total
    if total_rules > 0:
        results['overall_accuracy'] = correct_rules / total_rules

    # per-rule accuracies
    for r in short_rules:
        entry = results['rule_accuracy'][r]
        if entry['total'] > 0:
            entry['accuracy'] = entry['correct'] / entry['total']
        else:
            entry['accuracy'] = 0.0

    # add summary counts
    results['counts']['correct_persons'] = correct_persons
    results['counts']['total_rule_checks'] = total_rules
    results['counts']['correct_rule_checks'] = correct_rules

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate verification results')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth JSON file')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--output', type=str, default='logs/metrics/evaluation_report.json',
                        help='Output path for evaluation report (defaults to logs/metrics/...)')

    args = parser.parse_args()

    # Load files (robustly)
    try:
        ground_truth = _load_json_file_with_fallback(args.ground_truth)
    except Exception as e:
        print(f"Error loading ground truth file '{args.ground_truth}': {e}", file=sys.stderr)
        sys.exit(2)

    try:
        predictions = _load_json_file_with_fallback(args.predictions)
    except Exception as e:
        print(f"Error loading predictions file '{args.predictions}': {e}", file=sys.stderr)
        sys.exit(2)

    # Evaluate
    results = evaluate_predictions(ground_truth, predictions)

    # Save results (ensure parent exists)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write evaluation report to '{out_path}': {e}", file=sys.stderr)
        sys.exit(3)

    # Print summary
    print(f"Evaluation completed. Results saved to: {out_path}")
    print(f"Overall Accuracy (rule-level): {results['overall_accuracy']:.2%}")
    print(f"Person-level Accuracy: {results['person_level_accuracy']:.2%}")
    print(f"GT count: {results['counts'].get('gt_count')}, Predictions count: {results['counts'].get('pred_count')}")
    print(f"Matched persons: {results['counts'].get('matched_persons')}, Missing predictions: {results['counts'].get('missing_predictions')}")

    # Optionally print top few details for quick debugging
    if results.get('details'):
        print("\nSample mismatches (first 5):")
        for d in results['details'][:5]:
            print(json.dumps(d, ensure_ascii=False))

if __name__ == "__main__":
    main()
