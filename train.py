import pandas as pd
import os
import pickle
from sklearn.ensemble import random_forest
import argparse

def train(input_train, out_model='models/model.pkl', n_estimators=100, max_depth=None):
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    
    print(f"Loading training data from {input_train}...")
    df = pd.read_csv(input_train)
    
    # MEDV is the target variable
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    print(f"Training RandomForestRegressor model with n_estimators={n_estimators}, max_depth={max_depth}...")
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X, y)
    
    # Save the model
    with open(out_model, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Saved trained model to {out_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/train.csv', help='Input train CSV file')
    parser.add_argument('--model', default='models/model.pkl', help='Output model path')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    args = parser.parse_args()
    
    train(args.input, args.model, args.n_estimators, args.max_depth)
