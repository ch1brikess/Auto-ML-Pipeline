import pandas as pd
import numpy as np
import os
import json
import pickle
from pathlib import Path
import warnings
from colorama import init, Fore
init(autoreset=True)
warnings.filterwarnings('ignore')

class TestDataPreprocessor:
    def __init__(self, preprocessing_info_path, preprocessing_models_path):
        self.preprocessing_info = self.load_preprocessing_info(preprocessing_info_path)
        self.models = self.load_models(preprocessing_models_path)
        self.target_column = self.preprocessing_info['target_column']
        self.output_columns = self.preprocessing_info.get('output_columns', [])
        
        self.one_hot_encoded_columns = self.preprocessing_info.get('one_hot_encoded_columns', {})
        self.high_cardinality_columns = self.preprocessing_info.get('high_cardinality_columns', [])
        self.label_encoders_info = self.preprocessing_info.get('label_encoders', {})
        
    def load_preprocessing_info(self, info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_models(self, models_path):
        with open(models_path, 'rb') as f:
            return pickle.load(f)
    
    def ensure_feature_consistency(self, df, args):
        training_features = self.preprocessing_info.get('training_features', [])
        
        if training_features:
            output_data = df[self.output_columns].copy() if all(col in df.columns for col in self.output_columns) else pd.DataFrame()
            
            result_df = pd.DataFrame(index=df.index)
            
            for feature in training_features:
                if feature != self.target_column:
                    if feature in df.columns:
                        result_df[feature] = df[feature]
                    else:
                        result_df[feature] = 0
                        print(Fore.BLUE+f"Added missing feature: {feature}")
            
            for col in self.output_columns:
                if col in df.columns:
                    result_df[col] = df[col]
            
            print(Fore.BLUE+f"PassengerId preserved: {len(result_df[args.output_columns[0]])} values")
            print(Fore.BLUE+f"PassengerId sample: {result_df[args.output_columns[0]].head().tolist()}")
            return result_df
        
        return df
    
    def handle_missing_values(self, df):
        for col in df.columns:
            if col in self.output_columns:
                continue
                
            if col == self.target_column and self.target_column in df.columns:
                continue
                
            if df[col].isnull().any():
                if df[col].dtype in ['object', 'category']:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        return df
    
    def encode_categorical_features(self, df):
        label_encoders_info = self.preprocessing_info.get('label_encoders', {})
        one_hot_encoded_columns = self.preprocessing_info.get('one_hot_encoded_columns', {})
        high_cardinality_columns = self.preprocessing_info.get('high_cardinality_columns', [])
        
        for col in high_cardinality_columns:
            if col in df.columns and col not in self.output_columns:
                df = df.drop(columns=[col])
                print(Fore.BLUE+f"Removed high cardinality column: {col}")
        
        for col, expected_columns in one_hot_encoded_columns.items():
            if col in df.columns and col not in self.output_columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                
                for expected_col in expected_columns:
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0
                
                dummies = dummies[expected_columns]
                
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                print(Fore.BLUE+f"Encoded categorical feature: {col} with one-hot encoding")
        
        for col, encoder_info in label_encoders_info.items():
            if col in df.columns and col == self.target_column and col not in self.output_columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.classes_ = np.array(encoder_info['classes'])
                
                mask = df[col].isin(le.classes_)
                df.loc[~mask, col] = le.classes_[0]
                df[col] = le.transform(df[col])
                print(Fore.MAGENTA+f"Encoded target column: {col}")
        
        return df
    
    def scale_features(self, df, args):
        scaler = self.models.get('scaler')
        if scaler is not None:
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in args.output_columns and col != args.target]
            
            if numeric_cols:
                df_numeric = df[numeric_cols]
                df[numeric_cols] = scaler.transform(df_numeric)
                print(Fore.BLUE+f"Scaled {len(numeric_cols)} numerical features")
        
        return df
    
    def apply_pca(self, df):
        pca = self.models.get('pca')
        if pca is not None:
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in self.output_columns and col != self.target_column]
            
            if numeric_cols:
                pca_features = pca.transform(df[numeric_cols])
                pca_columns = [f'PC{i+1}' for i in range(pca_features.shape[1])]
                pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
                
                df = df.drop(columns=numeric_cols)
                df = pd.concat([df, pca_df], axis=1)
                print(Fore.BLUE+f"Applied PCA transformation")
        
        return df
    
    def apply_preprocessing(self, df, args):
        print(Fore.MAGENTA+"Applying preprocessing to test data...")
        print(Fore.MAGENTA+f"Initial shape: {df.shape}")
        print(Fore.MAGENTA+f"Output columns to preserve: {args.output_columns}")
        
        missing_output_cols = set(args.output_columns) - set(df.columns)
        if missing_output_cols:
            print(Fore.YELLOW+f"Warning: Missing output columns in test data: {missing_output_cols}")
        
        df = self.handle_missing_values(df)
        
        df = self.encode_categorical_features(df)
        
        df = self.ensure_feature_consistency(df, args)
        
        df = self.scale_features(df, args)
        
        df = self.apply_pca(df)
        
        print(Fore.MAGENTA+f"Final shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        for col in df.columns:
            if col in self.output_columns:
                continue
                
            if col == self.target_column and self.target_column in df.columns:
                continue
                
            if df[col].isnull().any():
                if df[col].dtype in ['object', 'category']:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        return df
    
    def encode_categorical_features(self, df):
        """Универсальное кодирование категориальных признаков"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in self.output_columns:
                continue
                
            if col == self.target_column and self.task_type == 'classification':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(Fore.BLUE+f"Encoded target column '{col}'")
                
            elif col != self.target_column:
                n_unique = df[col].nunique()
                
                if n_unique <= 10:
                    unique_values = df[col].unique()
                    self.one_hot_encoded_columns[col] = [f"{col}_{val}" for val in unique_values]
                    
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    print(Fore.BLUE+f"One-hot encoded '{col}' ({n_unique} categories)")
                    
                elif n_unique <= 50:
                    freq_map = df[col].value_counts().to_dict()
                    df[f'{col}_freq_encoded'] = df[col].map(freq_map)
                    df = df.drop(columns=[col])
                    print(Fore.BLUE+f"Frequency encoded '{col}' ({n_unique} categories)")
                    
                else:
                    freq_map = df[col].value_counts().to_dict()
                    df[f'{col}_freq_encoded'] = df[col].map(freq_map)
                    df = df.drop(columns=[col])
                    print(Fore.BLUE+f"Frequency encoded high-cardinality '{col}' ({n_unique} categories)")
        
        return df
    
    def scale_features(self, df, args):
        scaler = None
        if scaler is not None:
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in args.output_columns and col != args.target]
            
            if numeric_cols:
                df_numeric = df[numeric_cols]
                df[numeric_cols] = scaler.transform(df_numeric)
                print(Fore.BLUE+f"Scaled {len(numeric_cols)} numerical features")
        
        return df
    
    def apply_pca(self, df):
        pca = self.models.get('pca')
        if pca is not None:
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in self.output_columns and col != self.target_column]
            
            if numeric_cols:
                pca_features = pca.transform(df[numeric_cols])
                pca_columns = [f'PC{i+1}' for i in range(pca_features.shape[1])]
                pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
                
                df = df.drop(columns=numeric_cols)
                df = pd.concat([df, pca_df], axis=1)
                print(Fore.BLUE+f"Applied PCA transformation")
        
        return df
    
    def apply_preprocessing(self, df, args):
        print(Fore.MAGENTA+"Applying preprocessing to test data...")
        print(Fore.MAGENTA+f"Initial shape: {df.shape}")
        print(Fore.MAGENTA+f"Output columns to preserve: {args.output_columns}")
        
        missing_output_cols = set(args.output_columns) - set(df.columns)
        if missing_output_cols:
            print(Fore.YELLOW+f"Warning: Missing output columns in test data: {missing_output_cols}")
        
        df = self.handle_missing_values(df)
        
        df = self.encode_categorical_features(df)
        
        df = self.ensure_feature_consistency(df, args)
        
        df = self.scale_features(df, args)
        
        df = self.apply_pca(df)
        
        print(Fore.MAGENTA+f"Final shape: {df.shape}")
        return df

def run_test_preprocessing(args):
    try:
        df = pd.read_csv(args.path)
        print(Fore.BLUE+f"Loaded test dataset: {df.shape}")
        print(Fore.BLUE+f"Test data columns: {list(df.columns)}")
    except Exception as e:
        print(Fore.RED+f"Error loading test file: {e}")
        return False
    
    cache_dir = Path(__file__).parent.parent.parent / 'cache'
    train_cache_dir = cache_dir / 'train'
    
    info_path = train_cache_dir / 'preprocessing_info.json'
    models_path = train_cache_dir / 'preprocessing_models.pkl'
    
    if not info_path.exists() or not models_path.exists():
        print(Fore.RED+"Error: Preprocessing info not found. Run train_preload.py first.")
        return False
    
    preprocessor = TestDataPreprocessor(info_path, models_path)
    
    processed_df = preprocessor.apply_preprocessing(df, args)
    
    if args.output_columns[0] in processed_df.columns:
        print(Fore.BLUE+f"Final PassengerId check: {processed_df[args.output_columns[0]].notna().sum()} values")
        print(Fore.BLUE+f"Final PassengerId sample: {processed_df[args.output_columns[0]].head().tolist()}")
    
    test_cache_dir = cache_dir / 'test'
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    
    input_filename = Path(args.path).stem
    output_path = test_cache_dir / f"{input_filename}_processed.csv"
    processed_df.to_csv(output_path, index=False)
    
    print(Fore.MAGENTA+f"Processed test dataset saved to: {output_path}")
    
    report_path = test_cache_dir / f"{input_filename}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Test Dataset Preprocessing Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input file: {args.path}\n")
        f.write(f"Target variable: {preprocessor.target_column}\n")
        f.write(f"Output columns: {preprocessor.output_columns}\n")
        f.write(f"Initial shape: {df.shape}\n")
        f.write(f"Final shape: {processed_df.shape}\n")
        f.write(f"Applied transformations from train preprocessing\n")
    
    print(Fore.MAGENTA+f"Test report saved to: {report_path}")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Dataset Preprocessor')
    parser.add_argument('--path', '-p', type=str, required=True, help='Path to test CSV dataset')
    parser.add_argument('--target', type=str, help='Target column name (optional for test)')
    parser.add_argument('--classification', action='store_true', help='Classification task')
    parser.add_argument('--regression', action='store_true', help='Regression task')
    parser.add_argument('--output_columns', type=str, nargs='+', help='Output columns to preserve')
    
    args = parser.parse_args()
    run_test_preprocessing(args)

if __name__ == "__main__":
    main()