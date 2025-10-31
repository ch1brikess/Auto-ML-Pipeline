import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import warnings
from colorama import init, Fore

init(autoreset=True)
warnings.filterwarnings('ignore')

class ClassifierPreprocessor:
    def __init__(self, target_column, output_columns=None):
        self.target_column = target_column
        self.output_columns = output_columns or []
        self.label_encoders = {}
        self.one_hot_encoded_columns = {}
        self.high_cardinality_columns = []
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.selected_columns = None
        self.training_features = None
        
    def remove_unnecessary_columns(self, df):
        cols_to_remove = []
        
        for col in df.columns:
            if col in self.output_columns or col == self.target_column:
                continue
                
            if df[col].nunique() == len(df):
                cols_to_remove.append(col)
                continue
                
            if df[col].nunique() <= 1:
                cols_to_remove.append(col)
                continue
                
            if df[col].isnull().mean() > 0.5:
                cols_to_remove.append(col)
                continue
                
            if df[col].nunique() / len(df) > 0.9:
                cols_to_remove.append(col)
                continue
        
        if cols_to_remove:
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
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        return df
    
    def encode_categorical_features(self, df):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in self.output_columns:
                continue
                
            if col == self.target_column:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            elif col != self.target_column:
                n_unique = df[col].nunique()
                
                if n_unique <= 10:
                    unique_values = df[col].unique()
                    self.one_hot_encoded_columns[col] = sorted([f"{col}_{val}" for val in unique_values])
                    
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                else:
                    if n_unique / len(df) > 0.5:
                        df = df.drop(columns=[col])
                        self.high_cardinality_columns.append(col)
                    else:
                        freq_encoding = df[col].value_counts().to_dict()
                        df[col] = df[col].map(freq_encoding)
                        df[col] = df[col].fillna(0)
        
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
        selector = SelectKBest(score_func=f_classif, k=min(20, len(numeric_cols)))
        
        try:
            selector.fit(X_numeric, y)
            selected_features = numeric_cols[selector.get_support()]
            
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            final_columns = list(selected_features) + list(non_numeric_cols) + [self.target_column] + self.output_columns
            
            self.selected_columns = final_columns
            return df[final_columns]
        except Exception as e:
            return df
    
    def scale_features(self, df):
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in self.output_columns and col != self.target_column]
        
        if not numeric_cols:
            return df
        
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
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
            except Exception as e:
                pass
        
        return df
    
    def preprocess(self, df, is_training=True):
        df = self.remove_unnecessary_columns(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df)
        df = self.remove_linear_dependencies(df)
        
        if is_training:
            df = self.select_best_features(df)
        
        if is_training:
            self.training_features = [col for col in df.columns if col not in self.output_columns and col != self.target_column]
            df = self.scale_features(df)
            df = self.apply_pca(df)
        
        df = df.loc[:, ~df.columns.duplicated()]
        
        columns_to_remove = []
        for col in df.columns:
            if col != self.target_column and self.target_column in col:
                columns_to_remove.append(col)
        
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
        
        return df

def save_preprocessing_info(preprocessor, output_path):
    info = {
        'target_column': preprocessor.target_column,
        'output_columns': preprocessor.output_columns,
        'selected_columns': preprocessor.selected_columns,
        'training_features': preprocessor.training_features,
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

def run_classifier_preprocessing(args, is_training=True):
    try:
        df = pd.read_csv(args.path)
    except Exception as e:
        print(Fore.RED+f"Error loading file: {e}")
        return False
    
    if is_training and args.target not in df.columns:
        print(Fore.RED+f"Error: target column '{args.target}' not found in dataset")
        return False
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'cache' / ('train' if is_training else 'test')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = ClassifierPreprocessor(
        target_column=args.target,
        output_columns=args.output_columns or []
    )
    processed_df = preprocessor.preprocess(df, is_training=is_training)
    
    input_filename = Path(args.path).stem
    output_path = cache_dir / f"{input_filename}_processed.csv"
    processed_df.to_csv(output_path, index=False)
    
    if is_training:
        info_path, model_path = save_preprocessing_info(preprocessor, output_path)
    
    return True