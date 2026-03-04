import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os

def validate(input_test, model_path, out_metrics='metrics.json', out_plot='actual_vs_predicted.png'):
    print(f"Loading test data from {input_test}...")
    df = pd.read_csv(input_test)
    X_test = df.drop('MEDV', axis=1)
    y_test = df['MEDV']
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'r2': r2
    }
    
    with open(out_metrics, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics: MSE={mse:.4f}, R2={r2:.4f}")
    
    print("Generating plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual MEDV')
    plt.ylabel('Predicted MEDV')
    plt.title('Actual vs Predicted Boston House Prices')
    plt.tight_layout()
    plt.savefig(out_plot)
    print(f"Generated plot: {out_plot}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', default='data/test.csv')
    parser.add_argument('--model', default='models/model.pkl')
    parser.add_argument('--out-metrics', default='metrics.json')
    parser.add_argument('--out-plot', default='actual_vs_predicted.png')
    args = parser.parse_args()
    
    validate(args.test_data, args.model, args.out_metrics, args.out_plot)
