import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import warnings
from colorama import Fore

warnings.filterwarnings('ignore')

class BaseMLPipeline:
    def __init__(self, algorithm, task_type, target_column, save_model, output_columns=None):
        self.algorithm = algorithm
        self.task_type = task_type
        self.target_column = target_column
        self.output_columns = output_columns or []
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.save_model = save_model
        
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
            raise ValueError(Fore.RED+f"Unsupported task type: {self.task_type}")
            
        if self.algorithm not in algorithms[self.task_type]:
            raise ValueError(Fore.RED+f"Unsupported algorithm: {self.algorithm} for {self.task_type}")
            
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
        print(Fore.GREEN+"Aligning features between train and test...")
        
        train_features = [col for col in X_train.columns 
                        if col not in self.output_columns 
                        and col != self.target_column
                        and self.target_column not in col]
        
        test_features = [col for col in X_test.columns 
                        if col not in self.output_columns 
                        and col != self.target_column
                        and self.target_column not in col]
        
        print(Fore.BLUE+f"Train features: {len(train_features)}")
        print(Fore.BLUE+f"Test features: {len(test_features)}")
        
        common_features = list(set(train_features) & set(test_features))
        missing_in_test = list(set(train_features) - set(test_features))
        extra_in_test = list(set(test_features) - set(train_features))
        
        print(Fore.BLUE+f"Common features: {len(common_features)}")
        print(Fore.BLUE+f"Missing in test: {len(missing_in_test)}")
        print(Fore.BLUE+f"Extra in test: {len(extra_in_test)}")
        
        if missing_in_test:
            print(Fore.YELLOW+f"Missing features in test: {missing_in_test}")
        
        X_train_aligned = X_train.copy()
        X_test_aligned = X_test.copy()
        
        for feature in missing_in_test:
            X_test_aligned[feature] = 0
            print(Fore.BLUE+f"Added missing feature: {feature}")
        
        if extra_in_test:
            X_test_aligned = X_test_aligned.drop(columns=extra_in_test)
            print(Fore.BLUE+f"Removed extra features: {extra_in_test}")
        
        final_features = common_features + missing_in_test
        X_train_aligned = X_train_aligned[final_features + [col for col in X_train_aligned.columns if col in self.output_columns]]
        X_test_aligned = X_test_aligned[final_features + [col for col in X_test_aligned.columns if col in self.output_columns]]
        
        self.feature_names = final_features
            
        print(Fore.BLUE+f"Final train shape: {X_train_aligned.shape}")
        print(Fore.BLUE+f"Final test shape: {X_test_aligned.shape}")
        
        return X_train_aligned, X_test_aligned
    
    def train_model(self, X_train, y_train):
        print(Fore.BLUE+f"Training {self.algorithm}...")
        
        feature_columns = [col for col in X_train.columns 
                        if col not in self.output_columns 
                        and col != self.target_column
                        and self.target_column not in col]
        
        X_train_features = X_train[feature_columns]
        self.feature_names = feature_columns
        
        print(Fore.BLUE+f"Training on {len(self.feature_names)} features")
        
        if len(self.feature_names) == 0:
            raise ValueError(Fore.RED+"No features available for training")
        
        algorithm_class = self.get_algorithm_class()
        param_grid = self.get_hyperparameters()
        
        if param_grid:
            print(Fore.BLUE+"Performing hyperparameter tuning...")
            
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            
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
            
            print(Fore.BLUE+f"Best parameters: {self.best_params}")
            print(Fore.BLUE+f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            self.model = algorithm_class(random_state=42)
            with tqdm(total=1, desc="Training model") as pbar:
                self.model.fit(X_train_features, y_train)
                pbar.update(1)
            self.best_params = "Default parameters"
        
        if self.save_model:
            import pickle
            import datetime
            
            results_dir = "results"
            Path(results_dir).mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{results_dir}/{self.algorithm}_model_{timestamp}.pkl"
            
            with open(model_filename, 'wb') as f:
                pickle.dump(self.model, f)
            
            print(Fore.BLUE+f"Model saved to: {model_filename}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_test_aligned = self.align_test_features(X_test)
        
        with tqdm(total=1, desc="Making predictions") as pbar:
            y_pred = self.model.predict(X_test_aligned)
            pbar.update(1)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            print(Fore.MAGENTA+f"Accuracy: {accuracy:.4f}")
            print(Fore.MAGENTA+"\nClassification Report:")
            print(classification_report(y_test, y_pred))
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(Fore.MAGENTA+f"Mean Squared Error: {mse:.4f}")
            print(Fore.MAGENTA+f"R² Score: {r2:.4f}")
            return {'mse': mse, 'r2': r2}
    
    def align_test_features(self, X_test):
        """Выравнивает тестовые данные по тренировочным фичам"""
        if self.feature_names is None:
            return X_test
        
        X_test_aligned = X_test.copy()
        
        for feature in self.feature_names:
            if feature not in X_test_aligned.columns:
                X_test_aligned[feature] = 0
                print(Fore.YELLOW+f"Added missing feature in prediction: {feature}")
        
        extra_features = set(X_test_aligned.columns) - set(self.feature_names) - set(self.output_columns)
        if extra_features:
            X_test_aligned = X_test_aligned.drop(columns=list(extra_features))
            print(Fore.YELLOW+f"Removed extra features in prediction: {list(extra_features)}")
        
        X_test_aligned = X_test_aligned[self.feature_names]
        
        return X_test_aligned
    
    def predict(self, X):
        if self.model is None:
            raise ValueError(Fore.RED+"Model not trained yet")
        
        X_aligned = self.align_test_features(X)
        
        with tqdm(total=1, desc="Final predictions") as pbar:
            predictions = self.model.predict(X_aligned)
            pbar.update(1)
        
        return predictions