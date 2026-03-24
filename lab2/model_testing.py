#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import argparse
import json

def evaluate_model(data_dir='artifacts/data',
                   model_dir='artifacts/models',
                   report_dir='artifacts/reports'):

    os.makedirs(report_dir, exist_ok=True)
    
    model = joblib.load(f'{model_dir}/model.pkl')
    
    X_test = pd.read_csv(f'{data_dir}/test_processed.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').squeeze()
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"Model test accuracy is: {accuracy:.3f}")
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    with open(f'{report_dir}/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Report saved in {report_dir}/metrics.json")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='artifacts/data')
    parser.add_argument('--model-dir', type=str, default='artifacts/models')
    parser.add_argument('--report-dir', type=str, default='artifacts/reports')
    args = parser.parse_args()
    evaluate_model(args.data_dir, args.model_dir, args.report_dir)