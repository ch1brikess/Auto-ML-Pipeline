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
            if col in self.output_columns or col == self.target_column:
                continue
                
            if df[col].isnull().any():
                missing_percent = df[col].isnull().mean()
                
                if missing_percent > 0.8:
                    df = df.drop(columns=[col])
                    print(Fore.RED+f"Removed column '{col}' with {missing_percent:.1%} missing values")
                    
                elif missing_percent > 0.3:
                    df[f'{col}_is_missing'] = df[col].isnull().astype(int)
                    
                    if df[col].dtype in ['object', 'category']:
                        df[col] = df[col].fillna('Missing')
                    else:
                        df[col] = df[col].fillna(-999)  
                        
                else:
                    if df[col].dtype in ['object', 'category']:
                        df[col] = df[col].fillna('Unknown')
                    else:
                        fill_value = df[col].median() if df[col].notna().sum() > 0 else 0
                        df[col] = df[col].fillna(fill_value)
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
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
                
                if n_unique > 10 and is_training and self.target_column in df.columns:
                    target_mean = df.groupby(col)[self.target_column].mean()
                    df[f'{col}_target_encoded'] = df[col].map(target_mean)
                    df[f'{col}_target_encoded'] = df[f'{col}_target_encoded'].fillna(df[self.target_column].mean())
                    df = df.drop(columns=[col])
                    print(Fore.BLUE+f"Applied target encoding to '{col}'")
                
                elif n_unique <= 15:
                    unique_values = df[col].unique()
                    self.one_hot_encoded_columns[col] = sorted([f"{col}_{val}" for val in unique_values])
                    
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    print(Fore.BLUE+f"Encoded '{col}' with one-hot encoding ({len(dummies.columns)} new features)")
                else:
                    freq_encoding = df[col].value_counts().to_dict()
                    df[f'{col}_freq'] = df[col].map(freq_encoding)
                    df = df.drop(columns=[col])
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
        
        if len(feature_columns) <= 30: 
            self.selected_columns = feature_columns + [self.target_column] + self.output_columns
            return df
        
        X = df[feature_columns]
        y = df[self.target_column]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df
            
        X_numeric = X[numeric_cols]
        
        k_features = max(int(len(numeric_cols) * 0.8), 30) 
        
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(k_features, len(numeric_cols)))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k_features, len(numeric_cols)))
        
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
        
        if len(numeric_cols) > 50:
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
    
    def create_new_features(self, df):
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 1:
            for i, col1 in enumerate(numeric_cols):
                if col1 in self.output_columns or col1 == self.target_column:
                    continue
                for col2 in numeric_cols[i+1:]:
                    if col2 not in self.output_columns and col2 != self.target_column:
                        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8) 
        
        for col in numeric_cols:
            if col in self.output_columns or col == self.target_column:
                continue
                
            if df[col].nunique() > 10:
                try:
                    df[f'{col}_bin'] = pd.qcut(df[col], 5, duplicates='drop', labels=False)
                except:
                    df[f'{col}_bin'] = pd.cut(df[col], 5, labels=False)
            
            if df[col].min() > 0 and df[col].skew() > 1:
                df[f'{col}_log'] = np.log1p(df[col])
            
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        
        for col in categorical_cols:
            if col in self.output_columns or col == self.target_column:
                continue
                
            n_unique = df[col].nunique()
            
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_freq'] = df[col].map(freq_map)
            
            if n_unique > 10:
                rare_categories = freq_map.keys()
                if len(rare_categories) > 0:
                    most_common = list(freq_map.keys())[0]
                    df[f'{col}_is_rare'] = (df[col] != most_common).astype(int)
        
        if len(numeric_cols) > 2:
            relevant_numeric = [col for col in numeric_cols 
                            if col not in self.output_columns and col != self.target_column]
            if relevant_numeric:
                df['row_mean'] = df[relevant_numeric].mean(axis=1)
                df['row_std'] = df[relevant_numeric].std(axis=1)
                df['row_sum'] = df[relevant_numeric].sum(axis=1)
        
        date_like_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                        ['date', 'time', 'year', 'month', 'day'])]
        
        for col in date_like_cols:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if not df[col].isnull().all():
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df[f'{col}_day'] = df[col].dt.day
                        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                except:
                    pass
        
        print(Fore.GREEN+f"Created {len(df.columns) - len(numeric_cols) - len(categorical_cols)} new features")
        return df
    
    def preprocess(self, df, is_training=True):
        print(Fore.GREEN+"Starting dataset processing...")
        print(Fore.BLUE+f"Initial shape: {df.shape}")
        
        df = self.create_new_features(df)
        
        print(Fore.BLUE+f"After feature engineering: {df.shape}")
        
        df = self.remove_unnecessary_columns(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df, is_training)  
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