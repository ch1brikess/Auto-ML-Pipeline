import pandas as pd
import numpy as np
import re
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
from colorama import init, Fore
init(autoreset=True)
warnings.filterwarnings('ignore')

class AdvancedTextPreprocessor:
    def __init__(self, text_columns=None, max_features=10000, method='tfidf', 
                 use_embeddings=False, language='english'):
        self.text_columns = text_columns or []
        self.max_features = max_features
        self.method = method
        self.use_embeddings = use_embeddings
        self.language = language
        self.vectorizers = {}
        self.scaler = None
        self.lda_models = {}
        self.svd_models = {}
        self.word_embeddings = {}
        
    def detect_text_columns(self, df):

        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':

                sample_size = min(100, len(df))
                sample_texts = df[col].dropna().sample(sample_size, random_state=42) if len(df[col].dropna()) > sample_size else df[col].dropna()
                
                if len(sample_texts) == 0:
                    continue
                    
                text_indicators = 0
                total_checked = 0
                
                for text in sample_texts:
                    text_str = str(text)
                    if len(text_str) > 10:
                        if re.search(r'\s+[a-zA-Zа-яА-Я]+\s+', ' ' + text_str + ' '):
                            text_indicators += 1
                        total_checked += 1
                
                if total_checked > 0 and text_indicators / total_checked > 0.3:
                    text_cols.append(col)
                    
        return text_cols
    
    def advanced_text_cleaning(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'\S+@\S+', '', text)
        

        text = re.sub(r'@\w+|#\w+', '', text)
        
        text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s\.\,\!\?\-\:]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        text = text.lower()
        
        return text
    
    def extract_text_features(self, text):
        if pd.isna(text) or text == "":
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'unique_words_ratio': 0,
                'stopword_ratio': 0,
                'digit_count': 0,
                'uppercase_ratio': 0,
                'special_char_ratio': 0,
                'emoticon_count': 0
            }
        
        text_str = str(text)
        

        char_count = len(text_str)
        words = text_str.split()
        word_count = len(words)
        

        sentence_count = max(1, text_str.count('.') + text_str.count('!') + text_str.count('?'))
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        unique_words_ratio = len(set(words)) / len(words) if words else 0
        

        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        stopword_count = sum(1 for word in words if word.lower() in common_words)
        stopword_ratio = stopword_count / len(words) if words else 0
        
        digit_count = sum(1 for char in text_str if char.isdigit())
        
        uppercase_count = sum(1 for char in text_str if char.isupper())
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
        

        special_chars = set('!@#$%^&*()_+-=[]{}|;:,.<>?/~`')
        special_char_count = sum(1 for char in text_str if char in special_chars)
        special_char_ratio = special_char_count / char_count if char_count > 0 else 0
        

        emoticon_patterns = [':)', ':(', ';)', ':D', ':/', ':P', ':|', ':*']
        emoticon_count = sum(text_str.count(pattern) for pattern in emoticon_patterns)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'unique_words_ratio': unique_words_ratio,
            'stopword_ratio': stopword_ratio,
            'digit_count': digit_count,
            'uppercase_ratio': uppercase_ratio,
            'special_char_ratio': special_char_ratio,
            'emoticon_count': emoticon_count
        }
    
    def create_ngram_features(self, texts, ngram_range=(1, 3)):
        if self.method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=ngram_range,
                stop_words='english' if self.language == 'english' else None,
                min_df=2,
                max_df=0.95
            )
        else:
            vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=ngram_range,
                stop_words='english' if self.language == 'english' else None,
                min_df=2,
                max_df=0.95
            )
        
        return vectorizer
    
    def preprocess_text_columns(self, df, is_training=True):
        if not self.text_columns:
            self.text_columns = self.detect_text_columns(df)
            print(Fore.BLUE + f"Auto-detected text columns: {self.text_columns}")
        
        processed_df = df.copy()
        
        for col in self.text_columns:
            if col in processed_df.columns:
                print(Fore.BLUE + f"Processing text column: {col}")
                
                processed_df[f'{col}_cleaned'] = processed_df[col].apply(self.advanced_text_cleaning)
                

                text_features_list = []
                for text in processed_df[f'{col}_cleaned']:
                    text_features_list.append(self.extract_text_features(text))
                
                text_features_df = pd.DataFrame(text_features_list, index=processed_df.index)
                

                for feature_col in text_features_df.columns:
                    processed_df[f'{col}_{feature_col}'] = text_features_df[feature_col]
                

                if is_training:
                    vectorizer = self.create_ngram_features(processed_df[f'{col}_cleaned'])
                    
                    try:
                        text_features = vectorizer.fit_transform(processed_df[f'{col}_cleaned'])
                        self.vectorizers[col] = vectorizer
                    except Exception as e:
                        print(Fore.RED + f"Error in vectorization for {col}: {e}")
                        continue
                else:
                    if col in self.vectorizers:
                        vectorizer = self.vectorizers[col]
                        text_features = vectorizer.transform(processed_df[f'{col}_cleaned'])
                    else:
                        print(Fore.YELLOW + f"No vectorizer found for {col}, skipping")
                        continue
                

                feature_names = [f'{col}_ngram_{i}' for i in range(text_features.shape[1])]
                text_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=processed_df.index)
                

                processed_df = pd.concat([processed_df, text_df], axis=1)
                
                print(Fore.GREEN + f"Created {text_features.shape[1]} n-gram features from {col}")
        
        return processed_df
    
    def extract_sentiment_features(self, df):
        for col in self.text_columns:
            if col in df.columns and f'{col}_cleaned' in df.columns:
                positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                                'perfect', 'love', 'best', 'beautiful', 'happy', 'nice', 'awesome'}
                negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
                                'dislike', 'angry', 'sad', 'poor', 'wrong', 'stupid'}
                
                df[f'{col}_positive_count'] = df[f'{col}_cleaned'].apply(
                    lambda x: sum(1 for word in str(x).split() if word in positive_words)
                )
                df[f'{col}_negative_count'] = df[f'{col}_cleaned'].apply(
                    lambda x: sum(1 for word in str(x).split() if word in negative_words)
                )
                df[f'{col}_sentiment_ratio'] = (
                    df[f'{col}_positive_count'] - df[f'{col}_negative_count']
                ) / (df[f'{col}_positive_count'] + df[f'{col}_negative_count'] + 1)
                

                df[f'{col}_exclamation_count'] = df[col].apply(lambda x: str(x).count('!'))
                df[f'{col}_question_count'] = df[col].apply(lambda x: str(x).count('?'))
                
        return df
    
    def apply_topic_modeling(self, df, n_topics=10, is_training=True):
        for col in self.text_columns:
            if col in df.columns and f'{col}_cleaned' in df.columns:
                text_feature_columns = [col for col in df.columns if f'{col}_ngram_' in col]
                
                if len(text_feature_columns) < 50:
                    continue
                    
                if is_training:
                    optimal_topics = min(n_topics, len(text_feature_columns) // 10)
                    optimal_topics = max(3, optimal_topics)  
                    
                    lda = LatentDirichletAllocation(
                        n_components=optimal_topics,
                        random_state=42,
                        max_iter=20,
                        learning_method='online'
                    )
                    
                    try:
                        lda_features = lda.fit_transform(df[text_feature_columns])
                        self.lda_models[col] = lda
                    except Exception as e:
                        print(Fore.RED + f"LDA failed for {col}: {e}")
                        continue
                else:
                    lda = self.lda_models.get(col)
                    if lda:
                        lda_features = lda.transform(df[text_feature_columns])
                    else:
                        continue
                
                topic_columns = [f'{col}_topic_{i}' for i in range(lda_features.shape[1])]
                topic_df = pd.DataFrame(lda_features, columns=topic_columns, index=df.index)
                df = pd.concat([df, topic_df], axis=1)
                
                print(Fore.GREEN + f"Added {lda_features.shape[1]} topics from {col}")
        
        return df
    
    def reduce_dimensionality(self, df, n_components=100, is_training=True):
        for col in self.text_columns:
            text_feature_columns = [col for col in df.columns if f'{col}_ngram_' in col]
            
            if len(text_feature_columns) > n_components:
                if is_training:

                    optimal_components = min(n_components, len(text_feature_columns) // 2)
                    optimal_components = max(20, optimal_components)  
                    
                    svd = TruncatedSVD(n_components=optimal_components, random_state=42)
                    reduced_features = svd.fit_transform(df[text_feature_columns])
                    self.svd_models[col] = svd
                    
                    explained_variance = svd.explained_variance_ratio_.sum()
                    print(Fore.BLUE + f"Explained variance for {col}: {explained_variance:.3f}")
                else:
                    svd = self.svd_models.get(col)
                    if svd:
                        reduced_features = svd.transform(df[text_feature_columns])
                    else:
                        continue
                

                columns_to_drop = [col for col in df.columns if f'{col}_ngram_' in col]
                df = df.drop(columns=columns_to_drop)
                
                svd_columns = [f'{col}_svd_{i}' for i in range(reduced_features.shape[1])]
                svd_df = pd.DataFrame(reduced_features, columns=svd_columns, index=df.index)
                df = pd.concat([df, svd_df], axis=1)
                
                print(Fore.GREEN + f"Reduced {len(text_feature_columns)} features to {reduced_features.shape[1]} for {col}")
        
        return df
    
    def create_interaction_features(self, df):
        text_stat_columns = []
        for col in self.text_columns:
            if col in df.columns:
                stat_cols = [col for col in df.columns if col.startswith(f'{col}_') and 
                           any(stat in col for stat in ['char_count', 'word_count', 'sentence_count', 
                                                       'avg_word_length', 'unique_words_ratio'])]
                text_stat_columns.extend(stat_cols)
        

        if len(text_stat_columns) >= 2:
            for i, col1 in enumerate(text_stat_columns):
                for col2 in text_stat_columns[i+1:]:
                    if col1 != col2:

                        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                        if (df[col2] != 0).all():
                            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        return df
    
    def preprocess(self, df, is_training=True):
        print(Fore.MAGENTA + "Starting advanced text preprocessing...")
        print(Fore.BLUE + f"Initial shape: {df.shape}")
        

        df = self.preprocess_text_columns(df, is_training)
        print(Fore.BLUE + f"After text preprocessing: {df.shape}")
        

        df = self.extract_sentiment_features(df)
        print(Fore.BLUE + f"After sentiment features: {df.shape}")
        

        if is_training:
            df = self.apply_topic_modeling(df, is_training=is_training)
            print(Fore.BLUE + f"After topic modeling: {df.shape}")
        
        df = self.reduce_dimensionality(df, is_training=is_training)
        print(Fore.BLUE + f"After dimensionality reduction: {df.shape}")
        
        df = self.create_interaction_features(df)
        print(Fore.BLUE + f"After interaction features: {df.shape}")
        
        columns_to_drop = [col for col in df.columns if '_cleaned' in col]
        df = df.drop(columns=columns_to_drop)
        
        print(Fore.MAGENTA + f"Final shape after text preprocessing: {df.shape}")
        return df
    
    def save_models(self, path):
        models = {
            'vectorizers': self.vectorizers,
            'lda_models': self.lda_models,
            'svd_models': self.svd_models,
            'text_columns': self.text_columns,
            'method': self.method,
            'max_features': self.max_features,
            'language': self.language
        }
        
        with open(path, 'wb') as f:
            pickle.dump(models, f)
        
        print(Fore.GREEN + f"Text preprocessing models saved to: {path}")
    
    def load_models(self, path):
        """Загрузка моделей"""
        with open(path, 'rb') as f:
            models = pickle.load(f)
        
        self.vectorizers = models.get('vectorizers', {})
        self.lda_models = models.get('lda_models', {})
        self.svd_models = models.get('svd_models', {})
        self.text_columns = models.get('text_columns', [])
        self.method = models.get('method', 'tfidf')
        self.max_features = models.get('max_features', 10000)
        self.language = models.get('language', 'english')
        
        print(Fore.GREEN + f"Text preprocessing models loaded from: {path}")

TextPreprocessor = AdvancedTextPreprocessor
