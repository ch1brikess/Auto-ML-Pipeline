import pandas as pd
import numpy as np
import zipfile
import os
import sys
import json
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'scr' / 'train'))
sys.path.append(str(Path(__file__).parent / 'scr' / 'test'))

def show_logo():
    logo = """[red]
        ╔║║╗╦╗╔╗╦║╔╔═╔╗╔╗
        ║╠╣║╠╣╠╝║╠╣╠═╚╗╚╗
        ╚║║╩╩╝╠╗╩║╚╚═╚╝╚╝
[/red]"""
    print(logo)

def show_about():
    about_file = Path(__file__).parent / 'about.txt'
    if about_file.exists():
        with open(about_file, 'r', encoding='utf-8') as f:
            about_text = f.read()
        print(about_text)
    else:
        print("ML Pipeline - Automated Machine Learning Pipeline")
        print("Version: 1.0")
        print("Description: Automated ML pipeline for classification and regression tasks")

class MLPipeline:
    def __init__(self, algorithm, task_type, target_column, output_columns=None):
        self.algorithm = algorithm
        self.task_type = task_type
        self.target_column = target_column
        self.output_columns = output_columns or []
        self.model = None
        self.best_params = None
        self.feature_names = None
        
    def get_algorithm_class(self):
        algorithms = {
            'classification': {
                'RandomForestClassifier': RandomForestClassifier,
                'LogisticRegression': LogisticRegression,
                'SVC': SVC,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'KNeighborsClassifier': KNeighborsClassifier
            },
            'regression': {
                'RandomForestRegressor': RandomForestRegressor,
                'LinearRegression': LinearRegression,
                'SVR': SVR,
                'DecisionTreeRegressor': DecisionTreeRegressor,
                'KNeighborsRegressor': KNeighborsRegressor
            }
        }
        
        if self.task_type not in algorithms:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
        if self.algorithm not in algorithms[self.task_type]:
            raise ValueError(f"Unsupported algorithm: {self.algorithm} for {self.task_type}")
            
        return algorithms[self.task_type][self.algorithm]
    
    def get_hyperparameters(self):
        if self.task_type == 'classification':
            if self.algorithm == 'RandomForestClassifier':
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.algorithm == 'LogisticRegression':
                return {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear']
                }
            elif self.algorithm == 'SVC':
                return {
                    'C': [0.1, 1],
                    'kernel': ['linear', 'rbf']
                }
            elif self.algorithm == 'DecisionTreeClassifier':
                return {
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.algorithm == 'KNeighborsClassifier':
                return {
                    'n_neighbors': [3, 5],
                    'weights': ['uniform', 'distance']
                }
        else:
            if self.algorithm == 'RandomForestRegressor':
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.algorithm == 'LinearRegression':
                return {
                    'fit_intercept': [True, False]
                }
            elif self.algorithm == 'SVR':
                return {
                    'C': [0.1, 1],
                    'kernel': ['linear', 'rbf']
                }
            elif self.algorithm == 'DecisionTreeRegressor':
                return {
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.algorithm == 'KNeighborsRegressor':
                return {
                    'n_neighbors': [3, 5],
                    'weights': ['uniform', 'distance']
                }
        
        return {}
    
    def align_features(self, X_train, X_test):
        print("Aligning features between train and test...")
        
        print(f"Train columns before alignment: {X_train.columns.tolist()}")
        print(f"Test columns before alignment: {X_test.columns.tolist()}")
        
        train_features = [col for col in X_train.columns 
                        if col not in self.output_columns 
                        and col != self.target_column
                        and self.target_column not in col]
        
        test_features = [col for col in X_test.columns 
                        if col not in self.output_columns 
                        and col != self.target_column
                        and self.target_column not in col]
        
        print(f"Train features (cleaned): {train_features}")
        print(f"Test features (cleaned): {test_features}")
        
        common_features = list(set(train_features) & set(test_features))
        missing_in_test = list(set(train_features) - set(test_features))
        extra_in_test = list(set(test_features) - set(train_features))
        
        print(f"Common features: {len(common_features)} - {common_features}")
        print(f"Missing in test: {len(missing_in_test)} - {missing_in_test}")
        print(f"Extra in test: {len(extra_in_test)} - {extra_in_test}")
        
        X_train_aligned = X_train.copy()
        X_test_aligned = X_test.copy()
        
        for feature in missing_in_test:
            X_test_aligned[feature] = 0
            print(f"Added missing feature: {feature}")
        
        features_to_keep = common_features + missing_in_test + [col for col in X_test_aligned.columns if col in self.output_columns]
        X_test_aligned = X_test_aligned[features_to_keep]
        
        final_train_features = train_features + [col for col in X_train_aligned.columns if col in self.output_columns]
        final_test_features = common_features + missing_in_test + [col for col in X_test_aligned.columns if col in self.output_columns]
        
        X_train_aligned = X_train_aligned[final_train_features]
        X_test_aligned = X_test_aligned[final_train_features]
        
        self.feature_names = train_features
            
        print(f"Train columns after alignment: {X_train_aligned.columns.tolist()}")
        print(f"Test columns after alignment: {X_test_aligned.columns.tolist()}")
        print(f"Train shape after alignment: {X_train_aligned.shape}")
        print(f"Test shape after alignment: {X_test_aligned.shape}")
        
        return X_train_aligned, X_test_aligned
    
    def train_model(self, X_train, y_train):
        print(f"Training {self.algorithm}...")
        
        feature_columns = [col for col in X_train.columns 
                        if col not in self.output_columns 
                        and col != self.target_column
                        and self.target_column not in col]
        
        X_train_features = X_train[feature_columns]
        
        self.feature_names = feature_columns
        
        print(f"Training on features: {self.feature_names}")
        print(f"Training data shape for model: {X_train_features.shape}")
        
        if len(self.feature_names) == 0:
            raise ValueError("No features available for training - possible data leakage issue")
        
        algorithm_class = self.get_algorithm_class()
        param_grid = self.get_hyperparameters()
        
        print(f"Target distribution in training data:")
        print(y_train.value_counts())
        
        if param_grid:
            print("Performing hyperparameter tuning...")
            
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            
            print(f"Testing {total_combinations} parameter combinations with 3-fold CV...")
            print(f"Total fits: {total_combinations * 3}")
            
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
            
            with tqdm(total=total_combinations * 3, desc="Hyperparameter tuning") as pbar:
                grid_search = GridSearchCV(
                    algorithm_class(random_state=42),
                    param_grid,
                    cv=3,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_features, y_train)
                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
                pbar.update(total_combinations * 3)
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            train_predictions = self.model.predict(X_train_features)
            print(f"Training predictions distribution: {pd.Series(train_predictions).value_counts()}")
            
            if grid_search.best_score_ > 0.95:
                print("⚠️  Warning: Very high CV score detected - possible data leakage")
        else:
            self.model = algorithm_class(random_state=42)
            with tqdm(total=1, desc="Training model") as pbar:
                self.model.fit(X_train_features, y_train)
                pbar.update(1)
            self.best_params = "Default parameters"
            
            train_predictions = self.model.predict(X_train_features)
            print(f"Training predictions distribution: {pd.Series(train_predictions).value_counts()}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.feature_names is not None:
            X_test_aligned = self.align_test_features(X_test)
        else:
            X_test_aligned = X_test
        
        with tqdm(total=1, desc="Making predictions") as pbar:
            y_pred = self.model.predict(X_test_aligned)
            pbar.update(1)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            return {'mse': mse, 'r2': r2}
    
    def align_test_features(self, X_test):
        for feature in self.feature_names:
            if feature not in X_test.columns:
                X_test[feature] = 0
        
        X_test = X_test[self.feature_names]
        return X_test
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.feature_names is not None:
            X_features = X[self.feature_names].copy()
        else:
            X_features = X.copy()
        
        with tqdm(total=1, desc="Final predictions") as pbar:
            predictions = self.model.predict(X_features)
            pbar.update(1)
        
        return predictions
    
def align_features(self, X_train, X_test):
    print("Aligning features between train and test...")
    
    print(f"Train columns before alignment: {X_train.columns.tolist()}")
    print(f"Test columns before alignment: {X_test.columns.tolist()}")
    
    common_features = list(set(X_train.columns) & set(X_test.columns))
    missing_in_test = list(set(X_train.columns) - set(X_test.columns))
    extra_in_test = list(set(X_test.columns) - set(X_train.columns))
    
    print(f"Common features: {len(common_features)}")
    print(f"Missing in test: {len(missing_in_test)} - {missing_in_test}")
    print(f"Extra in test: {len(extra_in_test)} - {extra_in_test}")
    
    for feature in missing_in_test:
        X_test[feature] = 0
        print(f"Added missing feature: {feature}")
    
    X_test = X_test[common_features + missing_in_test]
    
    X_test = X_test[X_train.columns]
    
    self.feature_names = X_train.columns.tolist()
    
    print(f"Train columns after alignment: {X_train.columns.tolist()}")
    print(f"Test columns after alignment: {X_test.columns.tolist()}")
    print(f"Train shape after alignment: {X_train.shape}")
    print(f"Test shape after alignment: {X_test.shape}")
    
    return X_train, X_test
    
    def train(self, X_train, y_train, X_test=None):
        print(f"Training {self.algorithm}...")
        
        self.feature_names = X_train.columns.tolist()
        
        algorithm_class = self.get_algorithm_class()
        param_grid = self.get_hyperparameters()
        
        print(f"Target distribution in training data:")
        print(y_train.value_counts())
        
        if param_grid:
            print("Performing hyperparameter tuning...")
            
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            
            print(f"Testing {total_combinations} parameter combinations with 3-fold CV...")
            print(f"Total fits: {total_combinations * 3}")
            
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
            
            grid_search = GridSearchCV(
                algorithm_class(random_state=42),
                param_grid,
                cv=3,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            train_predictions = self.model.predict(X_train)
            print(f"Training predictions distribution: {pd.Series(train_predictions).value_counts()}")
        else:
            self.model = algorithm_class(random_state=42)
            self.model.fit(X_train, y_train)
            self.best_params = "Default parameters"
            
            train_predictions = self.model.predict(X_train)
            print(f"Training predictions distribution: {pd.Series(train_predictions).value_counts()}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.feature_names is not None:
            X_test_aligned = self.align_test_features(X_test)
        else:
            X_test_aligned = X_test
        
        with tqdm(total=1, desc="Making predictions") as pbar:
            y_pred = self.model.predict(X_test_aligned)
            pbar.update(1)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            return {'mse': mse, 'r2': r2}
    
    def align_test_features(self, X_test):
        for feature in self.feature_names:
            if feature not in X_test.columns:
                X_test[feature] = 0
        
        X_test = X_test[self.feature_names]
        return X_test
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.feature_names is not None:
            X_aligned = self.align_test_features(X)
        else:
            X_aligned = X
        
        with tqdm(total=1, desc="Final predictions") as pbar:
            predictions = self.model.predict(X_aligned)
            pbar.update(1)
        
        return predictions

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(total=len(file_list), desc="Extracting files") as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                pbar.update(1)
    print("Extraction completed")

def find_csv_files(directory):
    csv_files = list(Path(directory).glob('*.csv'))
    return {file.stem: file for file in csv_files}

def find_submission_template(csv_files):
    for name, file in csv_files.items():
        if any(keyword in name.lower() for keyword in ['gender', 'submission', 'sample', 'template']):
            return file
    return None

def run_preprocessing(args):
    try:
        from train_preload import run_train_preprocessing
        from test_preload import run_test_preprocessing
        
        class TrainArgs:
            def __init__(self, path, target, classification, regression, output_columns):
                self.path = path
                self.target = target
                self.classification = classification
                self.regression = regression
                self.output_columns = output_columns or []
        
        with tqdm(total=2, desc="Data preprocessing") as pbar:
            print("Step 1/2: Preprocessing training data...")
            train_args = TrainArgs(
                path=str(Path('cashe/scr/train.csv')),
                target=args.target,
                classification=args.classification,
                regression=args.regression,
                output_columns=args.output_columns
            )
            
            if not run_train_preprocessing(train_args):
                return False
            pbar.update(1)
            
            print("Step 2/2: Preprocessing test data...")
            test_args = TrainArgs(
                path=str(Path('cashe/scr/test.csv')),
                target=args.target,
                classification=args.classification,
                regression=args.regression,
                output_columns=args.output_columns
            )
            
            if not run_test_preprocessing(test_args):
                return False
            pbar.update(1)
        
        return True
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    show_logo()
    
    parser = argparse.ArgumentParser(
        description='ML Pipeline Main Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -p data.zip -t Survived -c -a RandomForestClassifier -o PassengerId
  python main.py --path data.zip --target price -r --algorithm LinearRegression
  python main.py -h
  python main.py -a
        """
    )
    parser.add_argument('--path', '-p', type=str, help='Path to ZIP archive')
    parser.add_argument('--target', '-t', type=str, help='Target column name')
    parser.add_argument('--algorithm', '-a', type=str, 
                       help='Algorithm name (e.g., RandomForestClassifier, RandomForestRegressor, etc.)')
    parser.add_argument('--classification', '-c', action='store_true', help='Classification task')
    parser.add_argument('--regression', '-r', action='store_true', help='Regression task')
    parser.add_argument('--output_columns', '-o', type=str, nargs='+', help='Output columns to preserve')
    parser.add_argument('--about', '-ab', action='store_true', help='Show information about the program')
    
    args = parser.parse_args()
    
    if args.about:
        show_about()
        return
    
    if not all([args.path, args.target, args.algorithm]):
        parser.print_help()
        print("\n❌ Error: --path, --target, and --algorithm are required arguments")
        return
    
    if not (args.classification or args.regression):
        print("❌ Error: must specify --classification or --regression")
        return
    
    if args.classification and args.regression:
        print("❌ Error: cannot specify both --classification and --regression")
        return
    
    task_type = 'classification' if args.classification else 'regression'
    
    print("🚀 Starting ML Pipeline...")
    print(f"📁 Archive: {args.path}")
    print(f"🎯 Target: {args.target}")
    print(f"🤖 Algorithm: {args.algorithm}")
    print(f"📊 Task type: {task_type}")
    print("=" * 50)
    
    Path('cashe/scr').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)
    
    with tqdm(total=4, desc="Overall progress") as main_pbar:
        print("Step 1/4: Extracting archive...")
        try:
            extract_zip(args.path, 'cashe/scr')
        except Exception as e:
            print(f"❌ Error extracting zip file: {e}")
            return
        main_pbar.update(1)
        
        csv_files = find_csv_files('cashe/scr')
        print(f"📄 Found CSV files: {list(csv_files.keys())}")
        
        train_files = [f for name, f in csv_files.items() if 'train' in name.lower()]
        test_files = [f for name, f in csv_files.items() if 'test' in name.lower()]
        submission_file = find_submission_template(csv_files)
        
        train_file = train_files[0] if train_files else None
        test_file = test_files[0] if test_files else None
        
        if not train_file:
            print("❌ Error: train CSV file not found in archive")
            return
        
        print(f"✅ Train file: {train_file}")
        print(f"✅ Test file: {test_file}")
        
        if train_file:
            train_file.rename('cashe/scr/train.csv')
        if test_file:
            test_file.rename('cashe/scr/test.csv')
        
        if submission_file:
            submission_df = pd.read_csv(submission_file)
            print(f"📋 Found submission template: {submission_file.name}")
            print(f"📋 Template columns: {list(submission_df.columns)}")
            global_submission_template = submission_df
        else:
            global_submission_template = None
        
        print("Step 2/4: Data preprocessing...")
        try:
            success = run_preprocessing(args)
            if not success:
                print("❌ Error in preprocessing step")
                return
            main_pbar.update(1)
            print("✅ Preprocessing completed successfully")
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("Step 3/4: Loading processed data...")
        try:
            train_processed_path = Path('cashe/train/train_processed.csv')
            test_processed_path = Path('cashe/test/test_processed.csv')
            
            if not train_processed_path.exists():
                print(f"❌ Processed train file not found: {train_processed_path}")
                return
            if not test_processed_path.exists():
                print(f"❌ Processed test file not found: {test_processed_path}")
                return
            
            train_processed = pd.read_csv(train_processed_path)
            test_processed = pd.read_csv(test_processed_path)
            
            print(f"✅ Processed train data shape: {train_processed.shape}")
            print(f"✅ Processed test data shape: {test_processed.shape}")
            print(f"Train columns: {list(train_processed.columns)}")
            print(f"Test columns: {list(test_processed.columns)}")
            
            if 'PassengerId' in test_processed.columns:
                print(f"PassengerId in test data: {test_processed['PassengerId'].notna().sum()} values")
                print(f"First 5 PassengerId: {test_processed['PassengerId'].head().tolist()}")
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            import traceback
            traceback.print_exc()
            return
        main_pbar.update(1)
        
        if args.target not in train_processed.columns:
            print(f"❌ Error: target column '{args.target}' not found in processed data")
            print(f"Available columns: {list(train_processed.columns)}")
            return
        
        X_train = train_processed.drop(columns=[args.target])
        y_train = train_processed[args.target]
        
        if args.target in test_processed.columns:
            X_test = test_processed.drop(columns=[args.target])
            y_test = test_processed[args.target]
            has_test_target = True
        else:
            X_test = test_processed
            y_test = None
            has_test_target = False
        
        print("Step 4/4: Training model...")
        try:
            pipeline = MLPipeline(
                algorithm=args.algorithm,
                task_type=task_type,
                target_column=args.target,
                output_columns=args.output_columns or []
            )
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Test data shape: {X_test.shape}")
            
            print(f"Target distribution in training: {y_train.value_counts()}")
            
            print("Before alignment:")
            print(f"Train columns: {X_train.columns.tolist()}")
            print(f"Test columns: {X_test.columns.tolist()}")
            
            X_train_aligned, X_test_aligned = pipeline.align_features(X_train, X_test)
            
            print("After alignment:")
            print(f"Train columns: {X_train_aligned.columns.tolist()}")
            print(f"Test columns: {X_test_aligned.columns.tolist()}")
            
            print(f"Aligned training data shape: {X_train_aligned.shape}")
            print(f"Aligned test data shape: {X_test_aligned.shape}")
            
            print(f"NaN in training: {X_train_aligned.isnull().sum().sum()}")
            print(f"NaN in test: {X_test_aligned.isnull().sum().sum()}")
            
            pipeline.train_model(X_train_aligned, y_train)
            
            metrics = {}
            
            if has_test_target:
                print("\n📊 Evaluating on test data:")
                metrics = pipeline.evaluate(X_test_aligned, y_test)
            else:
                print("\n⚠️ No target in test data, skipping evaluation")
            
            print("\n🎯 Making predictions...")
            predictions = pipeline.predict(X_test_aligned)
            print(f"Predictions shape: {predictions.shape}")
            print(f"First 10 predictions: {predictions[:10]}")
            print(f"Predictions distribution: {pd.Series(predictions).value_counts()}")
            
            save_results(pipeline, predictions, test_processed, args, metrics, task_type)
            
            main_pbar.update(1)
            print("\n✅ Pipeline completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in ML pipeline: {e}")
            import traceback
            traceback.print_exc()

def save_results(pipeline, predictions, test_data, args, metrics, task_type):
    results_dir = Path('results')
    
    output_df = pd.DataFrame()
    
    passenger_id_col = None
    for col in ['PassengerId', 'passengerid', 'PassengerID', 'ID', 'Id']:
        if col in test_data.columns:
            passenger_id_col = col
            break
    
    if passenger_id_col:
        output_df['PassengerId'] = test_data[passenger_id_col].values
        print(f"✅ Using '{passenger_id_col}' as PassengerId with {len(test_data[passenger_id_col])} values")
        print(f"📊 PassengerId sample: {output_df['PassengerId'].head().tolist()}")
    else:
        output_df['PassengerId'] = range(1, len(predictions) + 1)
        print("⚠️ PassengerId not found, generated sequential IDs")
    
    if args.output_columns and len(args.output_columns) > 0:
        prediction_column = args.output_columns[-1]
        print(f"🎯 Using '{prediction_column}' as prediction column name")
    else:
        prediction_column = 'Survived'
        print(f"🎯 Using default '{prediction_column}' as prediction column name")
    
    output_df[prediction_column] = predictions
    
    if args.output_columns:
        for col in args.output_columns:
            if (col in test_data.columns and 
                col != 'PassengerId' and 
                col != prediction_column and
                col != passenger_id_col):
                output_df[col] = test_data[col].values
                print(f"✅ Added output column: {col}")
    
    predictions_path = results_dir / 'predictions.csv'
    output_df.to_csv(predictions_path, index=False)
    
    results_info = {
        'algorithm': args.algorithm,
        'task_type': task_type,
        'target_column': args.target,
        'best_parameters': pipeline.best_params,
        'metrics': metrics,
        'output_columns': args.output_columns or [],
        'prediction_column': prediction_column,
        'test_data_shape': test_data.shape,
        'predictions_shape': predictions.shape,
        'feature_names': pipeline.feature_names
    }
    
    info_path = results_dir / 'training_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(results_info, f, indent=2, ensure_ascii=False)
    
    report_path = results_dir / 'training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ML Pipeline Training Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Task type: {task_type}\n")
        f.write(f"Target column: {args.target}\n")
        f.write(f"Prediction column: {prediction_column}\n")
        f.write(f"Best parameters: {pipeline.best_params}\n")
        f.write(f"Features used: {len(pipeline.feature_names) if pipeline.feature_names else 'Unknown'}\n")
        
        if metrics:
            f.write("\nMetrics:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\nPredictions saved to: {predictions_path}\n")
        f.write(f"Total predictions: {len(predictions)}\n")
        f.write(f"Output columns: {list(output_df.columns)}\n")
    
    print(f"\n💾 Results saved to {results_dir}:")
    print(f"   - Predictions: {predictions_path}")
    print(f"   - Training info: {info_path}")
    print(f"   - Report: {report_path}")
    print(f"   - Output shape: {output_df.shape}")
    print(f"   - Prediction column: {prediction_column}")
    print(f"   - First few predictions:")
    print(output_df.head(10))
    
    if len(output_df) == 0:
        print("❌ ERROR: Output file is empty!")
    else:
        print(f"✅ SUCCESS: Generated {len(output_df)} predictions with PassengerId and {prediction_column} columns")

if __name__ == "__main__":
    main()