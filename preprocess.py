import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse

def preprocess(input_csv, out_train='data/train.csv', out_test='data/test.csv'):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    
    # Load dataset
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # The target variable is 'MEDV' (Median value of owner-occupied homes)
    # We will split the dataset
    print("Splitting data into train and test sets...")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save to CSV
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    
    print(f"Saved {out_train} with {len(train_df)} rows")
    print(f"Saved {out_test} with {len(test_df)} rows")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='Boston-house-price-data.csv', help='Input CSV file')
    args = parser.parse_args()
    
    preprocess(args.input)
