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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from tqdm import tqdm
import warnings
from colorama import init, Fore
import pickle
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'scr' / 'train'))
sys.path.append(str(Path(__file__).parent / 'scr' / 'test'))

init(autoreset=True)

import sys
import os
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    print("Testing encoding...")
except UnicodeEncodeError:
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

def show_logo():
    print(Fore.RED+'================================')
    logo = """
        ╔║║╗╦╗╔╗╦║╔╔═╔╗╔╗
        ║╠╣║╠╣╠╝║╠╣╠═╚╗╚╗
        ╚║║╩╩╝╠╗╩║╚╚═╚╝╚╝
"""
    print(Fore.RED+logo)
    print(f" {Fore.BLUE+'ML Pipeline'} {Fore.RED+'|'} {Fore.MAGENTA+'by ch1brikess'}")
    print(Fore.RED+'================================')
    
def show_about():
    about_file = Path(__file__).parent / 'about.txt'
    if about_file.exists():
        with open(about_file, 'r', encoding='utf-8') as f:
            about_text = f.read()
        print(Fore.GREEN+about_text)
    else:
        print(Fore.BLUE+"ML Pipeline - Automated Machine Learning Pipeline")
        print(Fore.MAGENTA+"Version: 2.0")
        print(Fore.BLUE+"Description: Universal ML pipeline for classification and regression tasks")
        print(Fore.MAGENTA+"Features: Auto feature engineering, hyperparameter tuning, cross-validation")
    return True

class MLPipeline:
    def __init__(self, algorithm, task_type, target_column, save_model, output_columns=None, 
                 cv_folds=3, random_state=42, n_jobs=-1, no_view=False):
        self.algorithm = algorithm
        self.task_type = task_type
        self.target_column = target_column
        self.output_columns = output_columns or []
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.save_model = save_model
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.no_view = no_view
        
    def print_message(self, message, color=Fore.BLUE):
        if not self.no_view:
            print(color + message)
        
    def get_algorithm_class(self):
        algorithms = {
            'classification': {
                'RandomForestClassifier': RandomForestClassifier,
                'LogisticRegression': LogisticRegression,
                'SVC': SVC,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'KNeighborsClassifier': KNeighborsClassifier,
                'GradientBoostingClassifier': GradientBoostingClassifier
            },
            'regression': {
                'RandomForestRegressor': RandomForestRegressor,
                'LinearRegression': LinearRegression,
                'SVR': SVR,
                'DecisionTreeRegressor': DecisionTreeRegressor,
                'KNeighborsRegressor': KNeighborsRegressor,
                'GradientBoostingRegressor': GradientBoostingRegressor
            }
        }
        
        if self.task_type not in algorithms:
            raise ValueError(Fore.RED+f"Unsupported task type: {self.task_type}")
            
        if self.algorithm not in algorithms[self.task_type]:
            raise ValueError(Fore.RED+f"Unsupported algorithm: {self.algorithm} for {self.task_type}")
            
        return algorithms[self.task_type][self.algorithm]
    
    def get_hyperparameters(self):
        if self.task_type == 'classification':
            if self.algorithm == 'RandomForestClassifier':
                return {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False]
                }
            elif self.algorithm == 'LogisticRegression':
                return {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l2'],
                    'max_iter': [1000]
                }
            elif self.algorithm == 'SVC':
                return {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            elif self.algorithm == 'DecisionTreeClassifier':
                return {
                    'max_depth': [3, 5, 7, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 6],
                    'criterion': ['gini', 'entropy']
                }
            elif self.algorithm == 'KNeighborsClassifier':
                return {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                    'p': [1, 2]
                }
            elif self.algorithm == 'GradientBoostingClassifier':
                return {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                }
        
        else:  
            if self.algorithm == 'RandomForestRegressor':
                return {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False]
                }
            elif self.algorithm == 'LinearRegression':
                return {
                    'fit_intercept': [True, False],
                    'copy_X': [True]
                }
            elif self.algorithm == 'SVR':
                return {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 0.2]
                }
            elif self.algorithm == 'DecisionTreeRegressor':
                return {
                    'max_depth': [3, 5, 7, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 6],
                    'criterion': ['squared_error', 'friedman_mse']
                }
            elif self.algorithm == 'KNeighborsRegressor':
                return {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                    'p': [1, 2]
                }
            elif self.algorithm == 'GradientBoostingRegressor':
                return {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                }
        
        return {}
    
    def align_features(self, X_train, X_test):
        self.print_message("Aligning features between train and test...", Fore.GREEN)
        
        train_features = [col for col in X_train.columns 
                        if col not in self.output_columns 
                        and col != self.target_column]
        
        test_features = [col for col in X_test.columns 
                        if col not in self.output_columns 
                        and col != self.target_column]
        
        self.print_message(f"Train features: {len(train_features)}")
        self.print_message(f"Test features: {len(test_features)}")
        
        common_features = list(set(train_features) & set(test_features))
        missing_in_test = list(set(train_features) - set(test_features))
        extra_in_test = list(set(test_features) - set(train_features))
        
        self.print_message(f"Common features: {len(common_features)}")
        self.print_message(f"Missing in test: {len(missing_in_test)}")
        self.print_message(f"Extra in test: {len(extra_in_test)}")
        
        X_train_aligned = X_train.copy()
        X_test_aligned = X_test.copy()
        
        for feature in missing_in_test:
            X_test_aligned[feature] = 0
            self.print_message(f"Added missing feature: {feature}", Fore.YELLOW)
        
        features_to_keep = common_features + missing_in_test + [col for col in self.output_columns if col in X_test_aligned.columns]
        
        X_train_aligned = X_train_aligned[features_to_keep]
        X_test_aligned = X_test_aligned[features_to_keep]
        
        self.feature_names = common_features + missing_in_test
            
        self.print_message(f"Final train shape: {X_train_aligned.shape}")
        self.print_message(f"Final test shape: {X_test_aligned.shape}")
        self.print_message(f"Features for model: {len(self.feature_names)}")
        
        return X_train_aligned, X_test_aligned
    
    def train_model(self, X_train, y_train):
        print(f"Training {self.algorithm}")
        
        feature_columns = [col for col in X_train.columns 
                        if col not in self.output_columns 
                        and col != self.target_column]
        
        X_train_features = X_train[feature_columns]
        
        self.feature_names = feature_columns
        
        print(f"Training on {len(self.feature_names)} features")
        print(f"Training data shape: {X_train_features.shape}")
        
        if len(self.feature_names) == 0:
            raise ValueError("No features available for training")
        
        algorithm_class = self.get_algorithm_class()
        param_grid = self.get_hyperparameters()
        
        print("Target distribution:")
        print(y_train.value_counts() if self.task_type == 'classification' else f"Range: {y_train.min():.2f} - {y_train.max():.2f}")
        
        print("Calculating cross-validation score")
        base_model = algorithm_class(random_state=self.random_state)
        cv_scores = cross_val_score(base_model, X_train_features, y_train, cv=self.cv_folds, 
                                scoring='accuracy' if self.task_type == 'classification' else 'r2')
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if param_grid:
            print("Performing hyperparameter tuning")
            
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            
            print(f"Testing {total_combinations} parameter combinations with {self.cv_folds}-fold CV")
            print(f"Total fits: {total_combinations * self.cv_folds}")
            
            # Если слишком много комбинаций, используем RandomizedSearchCV
            if total_combinations * self.cv_folds > 500:
                print("Too many combinations, using RandomizedSearchCV with 50 iterations")
                from sklearn.model_selection import RandomizedSearchCV
                scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
                
                grid_search = RandomizedSearchCV(
                    algorithm_class(random_state=self.random_state),
                    param_grid,
                    n_iter=50,
                    cv=self.cv_folds,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state
                )
            else:
                scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
                grid_search = GridSearchCV(
                    algorithm_class(random_state=self.random_state),
                    param_grid,
                    cv=self.cv_folds,
                    scoring=scoring,
                    n_jobs=self.n_jobs
                )
            
            try:
                grid_search.fit(X_train_features, y_train)
                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
                
                print(f"Best parameters: {self.best_params}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")
                
            except KeyboardInterrupt:
                print("Hyperparameter tuning interrupted, using default parameters")
                self.model = algorithm_class(random_state=self.random_state)
                self.model.fit(X_train_features, y_train)
                self.best_params = "Default parameters (tuning interrupted)"
            
        else:
            self.model = algorithm_class(random_state=self.random_state)
            self.model.fit(X_train_features, y_train)
            self.best_params = "Default parameters"
        
        if self.save_model:
            import pickle
            import datetime
            
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{results_dir}/{self.algorithm}_model_{timestamp}.pkl"
            
            with open(model_filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'best_params': self.best_params,
                    'target_column': self.target_column,
                    'output_columns': self.output_columns
                }, f)
            
            print(f"Model saved to: {model_filename}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.feature_names is not None:
            X_test_features = X_test[self.feature_names].copy()
        else:
            X_test_features = X_test.copy()
        
        if self.no_view:
            y_pred = self.model.predict(X_test_features)
        else:
            with tqdm(total=1, desc="Making predictions") as pbar:
                y_pred = self.model.predict(X_test_features)
                pbar.update(1)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            self.print_message(f"Test Accuracy: {accuracy:.4f}", Fore.MAGENTA)
            if not self.no_view:
                self.print_message("\nClassification Report:", Fore.MAGENTA)
                print(classification_report(y_test, y_pred))
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.print_message(f"Mean Squared Error: {mse:.4f}", Fore.MAGENTA)
            self.print_message(f"R² Score: {r2:.4f}", Fore.MAGENTA)
            return {'mse': mse, 'r2': r2}
    
    def predict(self, X):
        if self.model is None:
            raise ValueError(Fore.RED+"Model not trained yet")
        
        if self.feature_names is not None:
            X_features = X[self.feature_names].copy()
        else:
            X_features = X.copy()
        
        self.print_message(f"Predicting on {X_features.shape[1]} features")
        
        if self.no_view:
            predictions = self.model.predict(X_features)
        else:
            with tqdm(total=1, desc="Final predictions") as pbar:
                predictions = self.model.predict(X_features)
                pbar.update(1)
        
        return predictions
    
    def save_model_to_path(self, model_path):
        """Save model to specific path"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'best_params': self.best_params,
                'target_column': self.target_column,
                'output_columns': self.output_columns,
                'algorithm': self.algorithm,
                'task_type': self.task_type
            }, f)
        
        self.print_message(f"Model saved to: {model_path}")
    
    def load_model_from_path(self, model_path):
        """Load model from specific path"""
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data['best_params']
        self.target_column = model_data['target_column']
        self.output_columns = model_data['output_columns']
        self.algorithm = model_data['algorithm']
        self.task_type = model_data['task_type']
        
        self.print_message(f"Model loaded from: {model_path}")
        self.print_message(f"Algorithm: {self.algorithm}")
        self.print_message(f"Task type: {self.task_type}")

def extract_zip(zip_path, extract_to):
    print(Fore.BLUE+f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(total=len(file_list), desc="Extracting files") as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                pbar.update(1)
    print(Fore.MAGENTA+"Extraction completed")

def find_csv_files(directory):
    csv_files = list(Path(directory).glob('*.csv'))
    return {file.stem: file for file in csv_files}

def find_submission_template(csv_files):
    for name, file in csv_files.items():
        if any(keyword in name.lower() for keyword in ['gender', 'submission', 'sample', 'template']):
            return file
    return None

def clear_directories(paths):
    for path in paths:
        folder_path = Path(path)
        if folder_path.exists():
            for filename in os.listdir(folder_path):
                file_path = folder_path / filename
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(Fore.RED+f'Error deleting file: {file_path}. {e}.')
    print(Fore.MAGENTA+'Cache cleared successfully.')

def run_preprocessing(args):
    try:
        from scr.train.train_preload import run_train_preprocessing
        from scr.test.test_preload import run_test_preprocessing
        
        class TrainArgs:
            def __init__(self, path, target, classification, regression, output_columns, 
                         feature_engineering=True, handle_missing=True, encode_categorical=True):
                self.path = path
                self.target = target
                self.classification = classification
                self.regression = regression
                self.output_columns = output_columns or []
                self.feature_engineering = feature_engineering
                self.handle_missing = handle_missing
                self.encode_categorical = encode_categorical
        
        if not args.only_test:
            print(Fore.RED+"Step 1/2: Preprocessing training data...")
            train_args = TrainArgs(
                path=str(Path('cache/scr/train.csv')),
                target=args.target,
                classification=args.classification,
                regression=args.regression,
                output_columns=args.output_columns,
                feature_engineering=args.feature_engineering,
                handle_missing=args.handle_missing,
                encode_categorical=args.encode_categorical
            )
            
            if not run_train_preprocessing(train_args):
                return False
        
        if not args.only_train:
            print(Fore.RED+"Step 2/2: Preprocessing test data...")
            test_args = TrainArgs(
                path=str(Path('cache/scr/test.csv')),
                target=args.target,
                classification=args.classification,
                regression=args.regression,
                output_columns=args.output_columns,
                feature_engineering=args.feature_engineering,
                handle_missing=args.handle_missing,
                encode_categorical=args.encode_categorical
            )
            
            if not run_test_preprocessing(test_args):
                return False
        
        return True
    except Exception as e:
        print(Fore.RED+f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    show_logo()
    
    parser = argparse.ArgumentParser(
        description='Universal ML Pipeline - Automated Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using ZIP archive
  python main.py --path data.zip --target Survived --classification --algorithm RandomForestClassifier --output_columns PassengerId
  
  # Using separate train/test files
  python main.py --train train.csv --test test.csv --target price --regression --algorithm GradientBoostingRegressor --output_columns ID
  
  # Train only mode
  python main.py --train train.csv --target label --classification --algorithm LogisticRegression --only_train --save_model
  
  # Test only mode with pre-trained model
  python main.py --test test.csv --target label --classification --only_test --model_path model.pkl --output_columns ID
  
  # From cache without preprocessing
  python main.py --from_cache --target Survived --classification --algorithm RandomForestClassifier --output_columns PassengerId
  
  # Fast execution without feature engineering
  python main.py --path data.zip --target target --classification --algorithm LogisticRegression --no_feature_engineering --no_tuning --no_view
  
  # High precision with extensive tuning
  python main.py --train train.csv --test test.csv --target label --classification --algorithm GradientBoostingClassifier --cv_folds 5 --n_jobs 4

Supported Algorithms:
  Classification: RandomForestClassifier, LogisticRegression, SVC, DecisionTreeClassifier, 
                  KNeighborsClassifier, GradientBoostingClassifier
  Regression:     RandomForestRegressor, LinearRegression, SVR, DecisionTreeRegressor, 
                  KNeighborsRegressor, GradientBoostingRegressor
        """
    )
    
    data_group = parser.add_argument_group('Data Input')
    data_group.add_argument('--path', '-p', type=str, help='Path to ZIP archive containing train/test data')
    data_group.add_argument('--train', type=str, help='Path to train CSV file')
    data_group.add_argument('--test', type=str, help='Path to test CSV file')
    data_group.add_argument('--target', '-t', type=str, help='Target column name')
    data_group.add_argument('--output_columns', '-o', type=str, nargs='+', help='Output columns to preserve (e.g., ID columns)')
    data_group.add_argument('--from_cache', action='store_true', help='Use already processed data from cache')
    
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--algorithm', '-a', type=str,
                           choices=['RandomForestClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier', 
                                   'KNeighborsClassifier', 'GradientBoostingClassifier',
                                   'RandomForestRegressor', 'LinearRegression', 'SVR', 'DecisionTreeRegressor',
                                   'KNeighborsRegressor', 'GradientBoostingRegressor'],
                           help='Machine learning algorithm to use')
    model_group.add_argument('--classification', '-c', action='store_true', help='Classification task')
    model_group.add_argument('--regression', '-r', action='store_true', help='Regression task')
    model_group.add_argument('--cv_folds', type=int, default=3, help='Number of cross-validation folds (default: 3)')
    model_group.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility (default: 42)')
    model_group.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs (default: -1 for all cores)')
    
    mode_group = parser.add_argument_group('Mode Options')
    mode_group.add_argument('--only_train', action='store_true', help='Train only mode (no test predictions)')
    mode_group.add_argument('--only_test', action='store_true', help='Test only mode (use pre-trained model)')
    mode_group.add_argument('--model_path', type=str, help='Path to pre-trained model for test only mode')
    
    preprocess_group = parser.add_argument_group('Preprocessing Options')
    preprocess_group.add_argument('--no_feature_engineering', action='store_true', help='Disable automatic feature engineering')
    preprocess_group.add_argument('--no_handle_missing', action='store_true', help='Disable automatic missing value handling')
    preprocess_group.add_argument('--no_encode_categorical', action='store_true', help='Disable automatic categorical encoding')
    preprocess_group.add_argument('--no_tuning', action='store_true', help='Disable hyperparameter tuning')
    
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--save_model', '-sm', action='store_true', help='Save trained model to file')
    output_group.add_argument('--output_dir', type=str, default='results', help='Output directory for results (default: results)')
    output_group.add_argument('--no_view', action='store_true', help='Disable progress bars and detailed output')
    
    utility_group = parser.add_argument_group('Utility Options')
    utility_group.add_argument('--about', '-ab', action='store_true', help='Show information about the program')
    utility_group.add_argument('--clear', '-cl', action='store_true', help='Clear cache directories')
    utility_group.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.about:
        if show_about():
            return
    
    if args.clear:
        clear_directories(['cache/train', 'cache/test', 'cache/scr', 'results'])
        return
    
    if args.only_train and args.only_test:
        print(Fore.RED+"Error: cannot specify both --only_train and --only_test")
        return
    
    if args.only_test and not args.model_path:
        print(Fore.RED+"Error: --model_path is required for --only_test mode")
        return
    
    if args.only_test and (args.algorithm or args.classification or args.regression):
        print(Fore.YELLOW+"Warning: algorithm and task type will be loaded from model file in --only_test mode")
    
    if not args.from_cache and not args.only_test:
        if not (args.path or args.train):
            print(Fore.RED+"Error: must specify either --path, --train, or --from_cache")
            return
    
    if args.only_test and not args.test:
        print(Fore.RED+"Error: --test is required for --only_test mode")
        return
    
    if not args.only_test:
        if not args.target:
            print(Fore.RED+"Error: --target is required for training")
            return
        
        if not args.algorithm:
            print(Fore.RED+"Error: --algorithm is required for training")
            return
        
        if not (args.classification or args.regression):
            if 'Classifier' in args.algorithm:
                args.classification = True
            elif 'Regressor' in args.algorithm:
                args.regression = True
            else:
                print(Fore.RED+"Error: must specify --classification or --regression")
                return
        
        if args.classification and args.regression:
            print(Fore.RED+"Error: cannot specify both --classification and --regression")
            return
    
    args.feature_engineering = not args.no_feature_engineering
    args.handle_missing = not args.no_handle_missing
    args.encode_categorical = not args.no_encode_categorical
    
    if args.only_test:
        task_type = 'classification' 
    else:
        task_type = 'classification' if args.classification else 'regression'
    
    if not args.no_view:
        print(Fore.MAGENTA+"Starting ML Pipeline...")
        if args.only_train:
            print(Fore.YELLOW+"MODE: TRAIN ONLY")
        elif args.only_test:
            print(Fore.YELLOW+"MODE: TEST ONLY")
        else:
            print(Fore.YELLOW+"MODE: FULL PIPELINE")
        
        if not args.only_test:
            print(Fore.BLUE+f"Target: {args.target}")
            print(Fore.BLUE+f"Algorithm: {args.algorithm}")
            print(Fore.BLUE+f"Task type: {task_type}")
        
        print(Fore.BLUE+f"CV folds: {args.cv_folds}")
        print(Fore.BLUE+f"Random state: {args.random_state}")
        print(Fore.BLUE+f"Parallel jobs: {args.n_jobs}")
        
        if args.path:
            print(Fore.BLUE+f"Archive: {args.path}")
        if args.train:
            print(Fore.BLUE+f"Train file: {args.train}")
        if args.test:
            print(Fore.BLUE+f"Test file: {args.test}")
        if args.from_cache:
            print(Fore.BLUE+"Using cached data")
        if args.model_path:
            print(Fore.BLUE+f"Model path: {args.model_path}")
        
        if args.output_columns:
            print(Fore.BLUE+f"Output columns: {args.output_columns}")
        
        print(Fore.YELLOW+f"Feature engineering: {'ENABLED' if args.feature_engineering else 'DISABLED'}")
        print(Fore.YELLOW+f"Hyperparameter tuning: {'ENABLED' if not args.no_tuning else 'DISABLED'}")
        print("=" * 60)
    
    Path('cache/scr').mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.only_test:
            pipeline = MLPipeline(
                algorithm="", 
                task_type="",   
                target_column=args.target or "", 
                output_columns=args.output_columns or [],
                save_model=False,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                no_view=args.no_view
            )
            
            pipeline.load_model_from_path(args.model_path)
            
            if not args.from_cache:
                print(Fore.MAGENTA+"Loading test data...")
                test_data = pd.read_csv(args.test)
                print(Fore.BLUE+f"Test data shape: {test_data.shape}")
            else:
                test_processed_path = Path('cache/test/test_processed.csv')
                if not test_processed_path.exists():
                    print(Fore.RED+f"Processed test file not found: {test_processed_path}")
                    return
                test_data = pd.read_csv(test_processed_path)
                print(Fore.BLUE+f"Loaded cached test data: {test_data.shape}")
            
            print(Fore.MAGENTA+"Making predictions...")
            predictions = pipeline.predict(test_data)
            
            save_results(pipeline, predictions, test_data, args, {}, pipeline.task_type)
            
            print(Fore.MAGENTA+"\n" + "="*60)
            print(Fore.GREEN+"Test only mode completed successfully!")
            
        else:
            with tqdm(total=4, desc="Overall progress") as main_pbar:
                if not args.from_cache:
                    if args.path:
                        print(Fore.MAGENTA+"Step 1/4: Extracting archive...")
                        try:
                            extract_zip(args.path, 'cache/scr')
                        except Exception as e:
                            print(Fore.RED+f"Error extracting zip file: {e}")
                            return
                        
                        csv_files = find_csv_files('cache/scr')
                        print(Fore.BLUE+f"Found CSV files: {list(csv_files.keys())}")
                        
                        train_files = [f for name, f in csv_files.items() if 'train' in name.lower()]
                        test_files = [f for name, f in csv_files.items() if 'test' in name.lower()]
                        
                        train_file = train_files[0] if train_files else None
                        test_file = test_files[0] if test_files else None
                        
                        if not train_file:
                            print(Fore.RED+"Error: train CSV file not found in archive")
                            return
                        
                        print(Fore.MAGENTA+f"Train file: {train_file}")
                        if test_file:
                            print(Fore.MAGENTA+f"Test file: {test_file}")
                        
                        if train_file:
                            train_file.rename('cache/scr/train.csv')
                        if test_file and not args.only_train:
                            test_file.rename('cache/scr/test.csv')
                    else:
                        print(Fore.MAGENTA+"Step 1/4: Copying CSV files...")
                        import shutil
                        try:
                            shutil.copy2(args.train, 'cache/scr/train.csv')
                            if not args.only_train and args.test:
                                shutil.copy2(args.test, 'cache/scr/test.csv')
                            print(Fore.BLUE+"Files copied successfully")
                        except Exception as e:
                            print(Fore.RED+f"Error copying files: {e}")
                            return
                else:
                    print(Fore.MAGENTA+"Step 1/4: Using cached data...")
                    if not Path('cache/scr/train.csv').exists():
                        print(Fore.RED+"Error: No cached train data found")
                        return
                
                main_pbar.update(1)
                
                if not args.from_cache:
                    print(Fore.MAGENTA+"Step 2/4: Data preprocessing...")
                    try:
                        success = run_preprocessing(args)
                        if not success:
                            print(Fore.RED+"Error in preprocessing step")
                            return
                        main_pbar.update(1)
                        print(Fore.BLUE+"Preprocessing completed successfully")
                    except Exception as e:
                        print(Fore.RED+f"Error in preprocessing: {e}")
                        import traceback
                        traceback.print_exc()
                        return
                else:
                    print(Fore.MAGENTA+"Step 2/4: Skipping preprocessing (using cache)...")
                    main_pbar.update(1)
                
                print(Fore.MAGENTA+"Step 3/4: Loading processed data...")
                try:
                    train_processed_path = Path('cache/train/train_processed.csv')
                    
                    if not train_processed_path.exists():
                        print(Fore.RED+f"Processed train file not found: {train_processed_path}")
                        return
                    
                    train_processed = pd.read_csv(train_processed_path)
                    print(Fore.BLUE+f"Processed train data shape: {train_processed.shape}")
                    
                    if not args.only_train:
                        test_processed_path = Path('cache/test/test_processed.csv')
                        if not test_processed_path.exists():
                            print(Fore.RED+f"Processed test file not found: {test_processed_path}")
                            return
                        test_processed = pd.read_csv(test_processed_path)
                        print(Fore.BLUE+f"Processed test data shape: {test_processed.shape}")
                    
                except Exception as e:
                    print(Fore.RED+f"Error loading processed data: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                main_pbar.update(1)
                
                if args.target not in train_processed.columns:
                    print(Fore.RED+f"Error: target column '{args.target}' not found in processed data")
                    print(Fore.RED+f"Available columns: {list(train_processed.columns)}")
                    return
                
                X_train = train_processed.drop(columns=[args.target])
                y_train = train_processed[args.target]
                
                if not args.only_train:
                    if args.target in test_processed.columns:
                        X_test = test_processed.drop(columns=[args.target])
                        y_test = test_processed[args.target]
                        has_test_target = True
                    else:
                        X_test = test_processed
                        y_test = None
                        has_test_target = False
                
                print(Fore.MAGENTA+"Step 4/4: Training model...")
                try:
                    pipeline = MLPipeline(
                        algorithm=args.algorithm,
                        task_type=task_type,
                        target_column=args.target,
                        output_columns=args.output_columns or [],
                        save_model=args.save_model,
                        cv_folds=args.cv_folds,
                        random_state=args.random_state,
                        n_jobs=args.n_jobs,
                        no_view=args.no_view
                    )
                    
                    print(Fore.BLUE+f"Training data shape: {X_train.shape}")
                    
                    if args.only_train:
                        pipeline.train_model(X_train, y_train)
                        
                        if args.save_model:
                            model_path = f"{args.output_dir}/{args.algorithm}_model.pkl"
                            pipeline.save_model_to_path(model_path)
                        
                        print(Fore.MAGENTA+"\n" + "="*60)
                        print(Fore.GREEN+"Train only mode completed successfully!")
                        if args.save_model:
                            print(Fore.BLUE+f"Model saved to: {model_path}")
                    
                    else:
                        print(Fore.BLUE+f"Test data shape: {X_test.shape}")
                        
                        X_train_aligned, X_test_aligned = pipeline.align_features(X_train, X_test)
                        
                        print(Fore.BLUE+f"Aligned training data shape: {X_train_aligned.shape}")
                        print(Fore.BLUE+f"Aligned test data shape: {X_test_aligned.shape}")
                        
                        if args.no_tuning:
                            print(Fore.YELLOW+"Skipping hyperparameter tuning...")
                            algorithm_class = pipeline.get_algorithm_class()
                            pipeline.model = algorithm_class(random_state=args.random_state)
                            if args.no_view:
                                pipeline.model.fit(X_train_aligned[pipeline.feature_names], y_train)
                            else:
                                with tqdm(total=1, desc="Training model") as pbar:
                                    pipeline.model.fit(X_train_aligned[pipeline.feature_names], y_train)
                                    pbar.update(1)
                            pipeline.best_params = "Default parameters (no tuning)"
                        else:
                            pipeline.train_model(X_train_aligned, y_train)
                        
                        metrics = {}
                        if has_test_target:
                            print(Fore.BLUE+"\nEvaluating on test data:")
                            metrics = pipeline.evaluate(X_test_aligned, y_test)
                        else:
                            print(Fore.YELLOW+"\nNo target in test data, skipping evaluation")
                        
                        print(Fore.BLUE+"\nMaking predictions...")
                        predictions = pipeline.predict(X_test_aligned)
                        print(Fore.BLUE+f"Predictions shape: {predictions.shape}")
                        
                        save_results(pipeline, predictions, test_processed, args, metrics, task_type)
                        
                        main_pbar.update(1)
                        print(Fore.MAGENTA+"\n" + "="*60)
                        print(Fore.GREEN+"Pipeline completed successfully!")
                        print(Fore.MAGENTA+f"Results saved to '{args.output_dir}/' directory")
                    
                except Exception as e:
                    print(Fore.RED+f"Error in ML pipeline: {e}")
                    import traceback
                    traceback.print_exc()
    
    except Exception as e:
        print(Fore.RED+f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def save_results(pipeline, predictions, test_data, args, metrics, task_type):
    results_dir = Path(args.output_dir)
    
    output_df = pd.DataFrame()
    
    if args.output_columns:
        for col in args.output_columns:
            if col in test_data.columns:
                output_df[col] = test_data[col].values
                if not args.no_view:
                    print(Fore.BLUE+f"Added output column: {col}")
    
    if args.output_columns and len(args.output_columns) > 0:
        prediction_column = args.output_columns[-1] if task_type == 'classification' else 'prediction'
    else:
        prediction_column = 'prediction'
    
    output_df[prediction_column] = predictions
    
    predictions_path = results_dir / 'predictions.csv'
    output_df.to_csv(predictions_path, index=False)
    
    results_info = {
        'algorithm': pipeline.algorithm,
        'task_type': pipeline.task_type,
        'target_column': pipeline.target_column,
        'best_parameters': pipeline.best_params,
        'metrics': metrics,
        'output_columns': pipeline.output_columns,
        'prediction_column': prediction_column,
        'feature_names': pipeline.feature_names,
        'cv_folds': args.cv_folds,
        'random_state': args.random_state,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    info_path = results_dir / 'training_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(results_info, f, indent=2, ensure_ascii=False)
    
    report_path = results_dir / 'training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ML Pipeline Training Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Algorithm: {pipeline.algorithm}\n")
        f.write(f"Task type: {task_type}\n")
        f.write(f"Target column: {pipeline.target_column}\n")
        f.write(f"Best parameters: {pipeline.best_params}\n")
        f.write(f"Features used: {len(pipeline.feature_names)}\n")
        f.write(f"CV folds: {args.cv_folds}\n")
        
        if metrics:
            f.write("\nMetrics:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\nOutput saved to: {predictions_path}\n")
        f.write(f"Total predictions: {len(predictions)}\n")
        f.write(f"Output columns: {list(output_df.columns)}\n")
    
    if not args.no_view:
        print(Fore.MAGENTA+f"\nResults saved:")
        print(Fore.BLUE+f"   - Predictions: {predictions_path}")
        print(Fore.BLUE+f"   - Training info: {info_path}")
        print(Fore.BLUE+f"   - Report: {report_path}")
        print(Fore.BLUE+f"   - Output shape: {output_df.shape}")
        print(Fore.BLUE+f"   - Prediction column: {prediction_column}")
    
    if len(output_df) == 0:
        print(Fore.RED+"ERROR: Output file is empty!")
    else:
        if not args.no_view:
            print(Fore.GREEN+f"SUCCESS: Generated {len(output_df)} predictions")

if __name__ == "__main__":
    main()