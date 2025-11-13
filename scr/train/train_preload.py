import pandas as pd
import numpy as np
import os
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
import warnings
from colorama import init, Fore
init(autoreset=True)
warnings.filterwarnings('ignore')

class DatasetPreprocessor:
    def __init__(self, target_column, task_type='classification', output_columns=None):
        self.target_column = target_column
        self.task_type = task_type
        self.output_columns = output_columns or []
        self.label_encoders = {}
        self.one_hot_encoded_columns = {}
        self.high_cardinality_columns = []
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.selected_columns = None
        
    def ensure_feature_consistency(self, df, is_training=True):
        if is_training:
            features_to_use = [col for col in df.columns if col not in self.output_columns or col == self.target_column]
            self.training_features = [col for col in features_to_use if col != self.target_column]
            return df
        
        if hasattr(self, 'training_features'):
            output_data = df[self.output_columns] if all(col in df.columns for col in self.output_columns) else pd.DataFrame()
            
            feature_columns = [col for col in df.columns if col not in self.output_columns]
            df_features = df[feature_columns]
            
            missing_features = set(self.training_features) - set(df_features.columns)
            extra_features = set(df_features.columns) - set(self.training_features)
            
            for feature in missing_features:
                df_features[feature] = 0
                print(Fore.BLUE+f"Added missing feature: {feature}")
            
            df_features = df_features[self.training_features]
            
            if not output_data.empty:
                df = pd.concat([df_features, output_data], axis=1)
            else:
                df = df_features
            
            if missing_features:
                print(Fore.BLUE+f"Added {len(missing_features)} missing features")
            if extra_features:
                print(Fore.BLUE+f"Removed {len(extra_features)} extra features")
        
        return df
            
    def remove_unnecessary_columns(self, df):
        cols_to_remove = []
        
        for col in df.columns:
            if col in self.output_columns or col == self.target_column:
                continue
                
            if df[col].nunique() == len(df):
                cols_to_remove.append(col)
                print(Fore.BLUE+f"Removing column '{col}' with all unique values")
                continue
                
            if df[col].nunique() <= 1:
                cols_to_remove.append(col)
                print(Fore.BLUE+f"Removing column '{col}' with only one unique value")
                continue
                
            if df[col].isnull().mean() > 0.5:
                cols_to_remove.append(col)
                print(Fore.BLUE+f"Removing column '{col}' with more than 50% missing values")
                continue
                
            if df[col].nunique() / len(df) > 0.9:
                cols_to_remove.append(col)
                print(Fore.BLUE+f"Removing column '{col}' with high cardinality ({df[col].nunique()} unique values)")
                continue
        
        if cols_to_remove:
            print(Fore.BLUE+f"Removing unnecessary columns: {cols_to_remove}")
            df = df.drop(columns=cols_to_remove)
        
        return df
    
    def handle_missing_values(self, df):
        for col in df.columns:
            if col in self.output_columns:
                continue
                
            if col == self.target_column:
                continue
                
            if df[col].isnull().any():
                if df[col].dtype in ['object', 'category']:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                    print(Fore.RED+f"Filled missing values in '{col}' with mode: {mode_val}")
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(Fore.RED+f"Filled missing values in '{col}' with median: {median_val:.4f}")
        
        return df
    
    def encode_categorical_features(self, df):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in self.output_columns:
                continue
                
            if col == self.target_column and self.task_type == 'classification':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(Fore.BLUE+f"Encoded target column '{col}' with LabelEncoder")
            elif col != self.target_column:
                n_unique = df[col].nunique()
                
                if n_unique <= 10:
                    unique_values = df[col].unique()
                    self.one_hot_encoded_columns[col] = sorted([f"{col}_{val}" for val in unique_values])
                    
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    print(Fore.BLUE+f"Encoded '{col}' with one-hot encoding ({len(dummies.columns)} new features)")
                else:
                    if n_unique / len(df) > 0.5:
                        print(Fore.BLUE+f"Removing high cardinality column: {col} ({n_unique} unique values)")
                        df = df.drop(columns=[col])
                        self.high_cardinality_columns.append(col)
                    else:
                        freq_encoding = df[col].value_counts().to_dict()
                        df[col] = df[col].map(freq_encoding)
                        df[col] = df[col].fillna(0)
                        print(Fore.BLUE+f"Encoded '{col}' with frequency encoding ({n_unique} unique values)")
        
        return df
    
    def remove_linear_dependencies(self, df):
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in self.output_columns and col != self.target_column]
        
        if len(numeric_cols) < 2:
            return df
            
        corr_matrix = df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        if to_drop:
            print(Fore.BLUE+f"Removing linearly dependent features: {to_drop}")
            return df.drop(columns=to_drop)
        
        return df
    
    def select_best_features(self, df):
        if self.target_column not in df.columns:
            return df
            
        feature_columns = [col for col in df.columns if col not in self.output_columns and col != self.target_column]
        X = df[feature_columns]
        y = df[self.target_column]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df
            
        X_numeric = X[numeric_cols]
        
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(20, len(numeric_cols)))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(20, len(numeric_cols)))
        
        try:
            selector.fit(X_numeric, y)
            selected_features = numeric_cols[selector.get_support()]
            
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            final_columns = list(selected_features) + list(non_numeric_cols) + [self.target_column] + self.output_columns
            
            self.selected_columns = final_columns
            print(Fore.BLUE+f"Selected {len(selected_features)} best features out of {len(numeric_cols)} numeric features")
            return df[final_columns]
        except Exception as e:
            print(Fore.BLUE+f"Feature selection failed: {e}")
            return df
    
    def scale_features(self, df):
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in self.output_columns and col != self.target_column]
        
        if not numeric_cols:
            return df
        
        if self.task_type == 'classification':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        print(Fore.GREEN+f"Scaled {len(numeric_cols)} numeric features using {type(self.scaler).__name__}")
        return df
    
    def apply_pca(self, df):
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in self.output_columns and col != self.target_column]
        
        if len(numeric_cols) > 20:
            try:
                self.pca = PCA(n_components=0.95)
                pca_features = self.pca.fit_transform(df[numeric_cols])
                
                pca_columns = [f'PC{i+1}' for i in range(pca_features.shape[1])]
                pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
                
                df = df.drop(columns=numeric_cols)
                df = pd.concat([df, pca_df], axis=1)
                print(Fore.GREEN+f"Applied PCA: reduced {len(numeric_cols)} features to {pca_features.shape[1]} components")
            except Exception as e:
                print(Fore.RED+f"PCA failed: {e}")
        
        return df
    
    def preprocess(self, df, is_training=True):
        print(Fore.GREEN+"Starting dataset processing...")
        print(Fore.BLUE+f"Initial shape: {df.shape}")
        print(Fore.BLUE+f"Initial columns: {df.columns.tolist()}")
        
        df = self.remove_unnecessary_columns(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df)
        df = self.remove_linear_dependencies(df)
        
        if is_training:
            df = self.select_best_features(df)
        
        df = self.ensure_feature_consistency(df, is_training)
        
        df = df.loc[:, ~df.columns.duplicated()]
        
        columns_to_remove = []
        for col in df.columns:
            if col != self.target_column and self.target_column in col:
                columns_to_remove.append(col)
                print(Fore.BLUE+f"Removing column '{col}' - possible target leakage")
        
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
        
        if is_training:
            df = self.scale_features(df)
            df = self.apply_pca(df)
        
        print(Fore.MAGENTA+f"Final shape: {df.shape}")
        print(Fore.MAGENTA+f"Final columns: {df.columns.tolist()}")
        return df

def save_preprocessing_info(preprocessor, input_path, output_path):
    info = {
        'target_column': preprocessor.target_column,
        'task_type': preprocessor.task_type,
        'output_columns': preprocessor.output_columns,
        'selected_columns': preprocessor.selected_columns,
        'training_features': preprocessor.training_features if hasattr(preprocessor, 'training_features') else None,
        'label_encoders': {},
        'one_hot_encoded_columns': preprocessor.one_hot_encoded_columns,
        'high_cardinality_columns': preprocessor.high_cardinality_columns,
        'scaler_type': type(preprocessor.scaler).__name__ if preprocessor.scaler else None,
        'pca_components': preprocessor.pca.n_components_ if preprocessor.pca else None
    }
    
    for col, encoder in preprocessor.label_encoders.items():
        info['label_encoders'][col] = {
            'classes': encoder.classes_.tolist()
        }
    
    info_path = output_path.parent / 'preprocessing_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    model_path = output_path.parent / 'preprocessing_models.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'scaler': preprocessor.scaler,
            'pca': preprocessor.pca
        }, f)
    
    return info_path, model_path

def run_train_preprocessing(args):
    if not (args.classification or args.regression):
        print(Fore.RED+"Error: must specify --classification or --regression")
        return False
    
    if args.classification and args.regression:
        print(Fore.RED+"Error: cannot specify both --classification and --regression")
        return False
    
    task_type = 'classification' if args.classification else 'regression'
    
    try:
        df = pd.read_csv(args.path)
        print(Fore.BLUE+f"Loaded dataset: {df.shape}")
    except Exception as e:
        print(Fore.RED+f"Error loading file: {e}")
        return False
    
    if args.target not in df.columns:
        print(Fore.RED+f"Error: target column '{args.target}' not found in dataset")
        print(Fore.RED+f"Available columns: {list(df.columns)}")
        return False
    
    cache_dir = Path(__file__).parent.parent.parent / 'cache' / 'train'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = DatasetPreprocessor(
        target_column=args.target, 
        task_type=task_type,
        output_columns=args.output_columns or []
    )
    processed_df = preprocessor.preprocess(df)
    
    input_filename = Path(args.path).stem
    output_path = cache_dir / f"{input_filename}_processed.csv"
    processed_df.to_csv(output_path, index=False)
    
    print(Fore.BLUE+f"Processed dataset saved to: {output_path}")
    
    info_path, model_path = save_preprocessing_info(preprocessor, args.path, output_path)
    
    print(Fore.BLUE+f"Preprocessing info saved to: {info_path}")
    print(Fore.BLUE+f"Preprocessing models saved to: {model_path}")
    
    report_path = cache_dir / f"{input_filename}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Dataset Preprocessing Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input file: {args.path}\n")
        f.write(f"Task type: {task_type}\n")
        f.write(f"Target variable: {args.target}\n")
        f.write(f"Output columns: {args.output_columns or []}\n")
        f.write(f"Initial shape: {df.shape}\n")
        f.write(f"Final shape: {processed_df.shape}\n")
        f.write(f"Selected columns: {len(preprocessor.selected_columns) if preprocessor.selected_columns else 'All'}\n")
        f.write(f"Scaler used: {type(preprocessor.scaler).__name__ if preprocessor.scaler else 'None'}\n")
        f.write(f"PCA applied: {preprocessor.pca is not None}\n")
        f.write(f"High cardinality columns removed: {preprocessor.high_cardinality_columns}\n")
    
    print(Fore.RED+f"Report saved to: {report_path}")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Dataset Preprocessor')
    parser.add_argument('--path', '-p', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--classification', action='store_true', help='Classification task')
    parser.add_argument('--regression', action='store_true', help='Regression task')
    parser.add_argument('--output_columns', type=str, nargs='+', help='Output columns to preserve')
    
    args = parser.parse_args()
    run_train_preprocessing(args)

if __name__ == "__main__":
    main()