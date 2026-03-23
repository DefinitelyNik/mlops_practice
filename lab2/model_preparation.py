#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import argparse

def train_model(data_dir='artifacts/data', 
                model_dir='artifacts/models',
                model_type='logistic'):

    os.makedirs(model_dir, exist_ok=True)
    
    X_train = pd.read_csv(f'{data_dir}/train_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').squeeze()
    
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    print(f"Обучение модели {model_type}...")
    model.fit(X_train, y_train)
    
    model_path = f'{model_dir}/model.pkl'
    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='artifacts/data')
    parser.add_argument('--model-dir', type=str, default='artifacts/models')
    parser.add_argument('--model-type', type=str, default='logistic',
                       choices=['logistic', 'random_forest'])
    args = parser.parse_args()
    train_model(args.data_dir, args.model_dir, args.model_type)