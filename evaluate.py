#!/usr/bin/env python3
"""
Evaluation script for document verification pipeline.
"""

import argparse
import json
from typing import Dict, Any, List
from pathlib import Path

def evaluate_predictions(ground_truth: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
    """Evaluate prediction accuracy against ground truth."""
    
    # Convert to dict for easier lookup
    gt_dict = {gt['person_id']: gt for gt in ground_truth}
    pred_dict = {pred['person_id']: pred for pred in predictions}
    
    results = {
        'overall_accuracy': 0,
        'rule_accuracy': {},
        'person_level_accuracy': 0,
        'details': []
    }
    
    total_rules = 0
    correct_rules = 0
    correct_persons = 0
    
    # Initialize rule counters
    rule_names = [f'rule_{i}' for i in range(1, 8)]
    for rule in rule_names:
        results['rule_accuracy'][rule] = {'correct': 0, 'total': 0}
    
    for person_id, gt_data in gt_dict.items():
        if person_id not in pred_dict:
            continue
            
        pred_data = pred_dict[person_id]
        
        # Check person-level accuracy
        gt_status = gt_data.get('overall_status')
        pred_status = pred_data.get('overall_status')
        
        if gt_status == pred_status:
            correct_persons += 1
        
        # Check rule-level accuracy
        gt_rules = gt_data.get('verification_results', {})
        pred_rules = pred_data.get('verification_results', {})
        
        for rule_name in rule_names:
            if rule_name in gt_rules and rule_name in pred_rules:
                results['rule_accuracy'][rule_name]['total'] += 1
                total_rules += 1
                
                if (gt_rules[rule_name].get('status') == 
                    pred_rules[rule_name].get('status')):
                    results['rule_accuracy'][rule_name]['correct'] += 1
                    correct_rules += 1
    
    # Calculate accuracies
    if len(gt_dict) > 0:
        results['person_level_accuracy'] = correct_persons / len(gt_dict)
    
    if total_rules > 0:
        results['overall_accuracy'] = correct_rules / total_rules
        
        # Calculate per-rule accuracy
        for rule_name in rule_names:
            rule_data = results['rule_accuracy'][rule_name]
            if rule_data['total'] > 0:
                rule_data['accuracy'] = rule_data['correct'] / rule_data['total']
            else:
                rule_data['accuracy'] = 0
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate verification results')
    parser.add_argument('--ground_truth', type=str, required=True,
                       help='Path to ground truth JSON file')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--output', type=str, default='metrics/evaluation_report.json',
                       help='Output path for evaluation report')
    
    args = parser.parse_args()
    
    # Load files
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    # Evaluate
    results = evaluate_predictions(ground_truth, predictions)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to: {args.output}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Person-level Accuracy: {results['person_level_accuracy']:.2%}")

if __name__ == "__main__":
    main()