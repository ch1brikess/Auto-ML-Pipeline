import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from tqdm import tqdm
import warnings
from colorama import Fore

warnings.filterwarnings('ignore')

class NLPPipeline:
    def __init__(self, algorithm, target_column, task_type, save_model, output_columns=None):
        self.algorithm = algorithm
        self.target_column = target_column
        self.task_type = task_type
        self.output_columns = output_columns or []
        self.model = None
        self.vectorizer = None
        self.dimensionality_reduction = None
        self.best_params = None
        self.feature_names = None
        self.save_model = save_model
        
    def get_algorithm_class(self):
        algorithms = {
            'classification': {
                'RandomForestClassifier': RandomForestClassifier,
                'LogisticRegression': LogisticRegression
            },
            'regression': {
                'RandomForestRegressor': RandomForestRegressor
            },
            'nlp': {
                'RandomForestClassifier': RandomForestClassifier,
                'LogisticRegression': LogisticRegression,
                'RandomForestRegressor': RandomForestRegressor
            }
        }
        
        if self.task_type == 'nlp':
            if 'Classifier' in self.algorithm:
                actual_task_type = 'classification'
            elif 'Regressor' in self.algorithm:
                actual_task_type = 'regression'
            else:
                actual_task_type = 'classification'
        else:
            actual_task_type = self.task_type
            
        if actual_task_type not in algorithms:
            raise ValueError(Fore.RED+f"Unsupported task type: {actual_task_type}")
            
        if self.algorithm not in algorithms[actual_task_type]:
            raise ValueError(Fore.RED+f"Unsupported algorithm: {self.algorithm} for {actual_task_type}")
            
        return algorithms[actual_task_type][self.algorithm], actual_task_type
    
    def create_pipeline(self):
        algorithm_class, actual_task_type = self.get_algorithm_class()
        
        if actual_task_type == 'classification':
            if self.algorithm == 'LogisticRegression':
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ('svd', TruncatedSVD(n_components=100)),
                    ('clf', algorithm_class(random_state=42))
                ])
            else:
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=3000)),
                    ('clf', algorithm_class(random_state=42))
                ])
        else:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000)),
                ('clf', algorithm_class(random_state=42))
            ])
        
        return pipeline, actual_task_type
    
    def get_hyperparameters(self):
        algorithm_class, actual_task_type = self.get_algorithm_class()
        
        if actual_task_type == 'classification':
            if self.algorithm == 'LogisticRegression':
                return {
                    'tfidf__max_features': [2000, 5000],
                    'tfidf__ngram_range': [(1, 1), (1, 2)],
                    'clf__C': [0.1, 1, 10]
                }
            elif self.algorithm == 'RandomForestClassifier':
                return {
                    'tfidf__max_features': [2000, 5000],
                    'clf__n_estimators': [50, 100],
                    'clf__max_depth': [5, 10]
                }
        else:
            return {
                'tfidf__max_features': [2000, 5000],
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [5, 10]
            }
    
    def train_model(self, X_train, y_train):
        print(Fore.BLUE+f"Training {self.algorithm} for NLP...")
        
        pipeline, actual_task_type = self.create_pipeline()
        param_grid = self.get_hyperparameters()
        
        if param_grid:
            print(Fore.BLUE+"Performing hyperparameter tuning...")
            
            scoring = 'accuracy' if actual_task_type == 'classification' else 'r2'
            
            with tqdm(total=1, desc="Hyperparameter tuning") as pbar:
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=3,
                    scoring=scoring,
                    n_jobs=-27,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
                pbar.update(1)
            
            print(Fore.BLUE+f"Best parameters: {self.best_params}")
            print(Fore.BLUE+f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            self.model = pipeline
            with tqdm(total=1, desc="Training model") as pbar:
                self.model.fit(X_train, y_train)
                pbar.update(1)
            self.best_params = "Default parameters"
        
        if self.save_model:
            import pickle
            import datetime
            
            results_dir = "results"
            Path(results_dir).mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{results_dir}/{self.algorithm}_nlp_model_{timestamp}.pkl"
            
            with open(model_filename, 'wb') as f:
                pickle.dump(self.model, f)
            
            print(Fore.BLUE+f"Model saved to: {model_filename}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        with tqdm(total=1, desc="Making predictions") as pbar:
            y_pred = self.model.predict(X_test)
            pbar.update(1)
        
        algorithm_class, actual_task_type = self.get_algorithm_class()
        
        if actual_task_type == 'classification':
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
    
    def predict(self, X):
        if self.model is None:
            raise ValueError(Fore.RED+"Model not trained yet")
        
        with tqdm(total=1, desc="Final predictions") as pbar:
            predictions = self.model.predict(X)
            pbar.update(1)
        
        return predictions