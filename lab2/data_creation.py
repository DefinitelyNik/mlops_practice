#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, fetch_openml
import os
import argparse
import requests

def create_synthetic_data(n_samples=1000, output_dir='artifacts/data'):
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    print(f"Data saved: train={len(train_df)}, test={len(test_df)}")

def download_from_url(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Data downloaded in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='synthetic', 
                       choices=['synthetic', 'url', 'github'])
    parser.add_argument('--url', type=str, help='URL for data download')
    parser.add_argument('--output-dir', type=str, default='artifacts/data')
    args = parser.parse_args()
    
    if args.source == 'synthetic':
        create_synthetic_data(output_dir=args.output_dir)
    elif args.source == 'url' and args.url:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        download_from_url(args.url, f"{args.output_dir}/raw_data.csv")