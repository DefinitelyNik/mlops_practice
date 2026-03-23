#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse

def preprocess_data(data_dir='artifacts/data', output_dir='artifacts/data'):
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        f'{output_dir}/train_processed.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
        f'{output_dir}/test_processed.csv', index=False)
    
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    y_train.to_frame().to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_frame().to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print("Preprocessing has ended")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='artifacts/data')
    parser.add_argument('--output-dir', type=str, default='artifacts/data')
    args = parser.parse_args()
    preprocess_data(args.data_dir, args.output_dir)