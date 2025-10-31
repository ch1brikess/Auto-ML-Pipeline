import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import re
import string
from sklearn.preprocessing import LabelEncoder
import warnings
from colorama import init, Fore

init(autoreset=True)
warnings.filterwarnings('ignore')

class NLPPreprocessor:
    def __init__(self, target_column, text_column, output_columns=None):
        self.target_column = target_column
        self.text_column = text_column
        self.output_columns = output_columns or []
        self.label_encoder = None
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        text = text.lower()
        
        text = re.sub(r'http\S+', '', text)
        
        text = re.sub(r'@\w+', '', text)
        
        text = re.sub(r'#\w+', '', text)
        
        text = re.sub(r'\d+', '', text)
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def handle_missing_values(self, df):
        if self.text_column not in df.columns:
            raise ValueError(Fore.RED+f"Text column '{self.text_column}' not found in dataset")
        
        df[self.text_column] = df[self.text_column].fillna('')
        
        for col in df.columns:
            if col != self.text_column and col not in self.output_columns and col != self.target_column:
                if df[col].isnull().any():
                    if df[col].dtype in ['object', 'category']:
                        df[col] = df[col].fillna('Unknown')
                    else:
                        df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def encode_target(self, df):
        if self.target_column in df.columns:
            if df[self.target_column].dtype == 'object':
                self.label_encoder = LabelEncoder()
                df[self.target_column] = self.label_encoder.fit_transform(df[self.target_column])
        
        return df
    
    def preprocess(self, df, is_training=True):
        print(Fore.BLUE+f"Preprocessing NLP data: {df.shape}")
        
        df = self.handle_missing_values(df)
        print(Fore.BLUE+f"After handling missing values: {df.shape}")
        
        print(Fore.BLUE+"Cleaning text data...")
        df[self.text_column] = df[self.text_column].apply(self.clean_text)
        
        if is_training:
            df = self.encode_target(df)
            print(Fore.BLUE+f"After encoding target: {df.shape}")
        
        columns_to_keep = [self.text_column]
        if self.target_column in df.columns:
            columns_to_keep.append(self.target_column)
        if self.output_columns:
            columns_to_keep.extend([col for col in self.output_columns if col in df.columns])
        
        df = df[columns_to_keep]
        print(Fore.BLUE+f"Final processed shape: {df.shape}")
        
        return df

def save_preprocessing_info(preprocessor, output_path):
    info = {
        'target_column': preprocessor.target_column,
        'text_column': preprocessor.text_column,
        'output_columns': preprocessor.output_columns,
        'label_encoder_classes': preprocessor.label_encoder.classes_.tolist() if preprocessor.label_encoder else None
    }
    
    info_path = output_path.parent / 'nlp_preprocessing_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    return info_path

def run_nlp_preprocessing(args, is_training=True):
    try:
        df = pd.read_csv(args.path)
        print(Fore.BLUE+f"Loaded dataset: {df.shape}")
        print(Fore.BLUE+f"Columns: {list(df.columns)}")
    except Exception as e:
        print(Fore.RED+f"Error loading file: {e}")
        return False
    
    if is_training and args.target not in df.columns:
        print(Fore.RED+f"Error: target column '{args.target}' not found in dataset")
        return False
    
    if not hasattr(args, 'text_column') or not args.text_column:
        print(Fore.RED+"Error: text column not specified for NLP task")
        return False
    
    if args.text_column not in df.columns:
        print(Fore.RED+f"Error: text column '{args.text_column}' not found in dataset")
        return False
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'cache' / ('train' if is_training else 'test')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = NLPPreprocessor(
        target_column=args.target,
        text_column=args.text_column,
        output_columns=args.output_columns or []
    )
    processed_df = preprocessor.preprocess(df, is_training=is_training)
    
    input_filename = Path(args.path).stem
    output_path = cache_dir / f"{input_filename}_processed.csv"
    processed_df.to_csv(output_path, index=False)
    
    if is_training:
        info_path = save_preprocessing_info(preprocessor, output_path)
        print(Fore.BLUE+f"Preprocessing info saved to: {info_path}")
    
    print(Fore.GREEN+f"Processed data saved to: {output_path}")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='NLP Dataset Preprocessor')
    parser.add_argument('--path', '-p', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--target', type=str, help='Target column name (optional for test)')
    parser.add_argument('--text_column', type=str, required=True, help='Text column for NLP')
    parser.add_argument('--output_columns', type=str, nargs='+', help='Output columns to preserve')
    
    args = parser.parse_args()
    run_nlp_preprocessing(args)

if __name__ == "__main__":
    main()