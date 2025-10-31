import argparse
import sys
from pathlib import Path
from colorama import init, Fore
import pandas as pd
import traceback

init(autoreset=True)

sys.path.append(str(Path(__file__).parent / 'scr'))

from scr.utils.data_utils import show_logo, show_about, extract_zip, find_csv_files, clear_directories, save_results
from scr.modules.classifier.classifier_preload import run_classifier_preprocessing
from scr.modules.regressor.regressor_preload import run_regressor_preprocessing
from scr.modules.nlp.nlp_preload import run_nlp_preprocessing
from scr.modules.classifier.classifier_pipeline import ClassifierPipeline
from scr.modules.regressor.regressor_pipeline import RegressorPipeline
from scr.modules.nlp.nlp_pipeline import NLPPipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description='ML Pipeline for Classification, Regression and NLP')
    
    parser.add_argument('--path', type=str, required=True, help='Path to dataset file or zip archive')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--algorithm', type=str, required=True, help='ML algorithm to use')
    parser.add_argument('--output_columns', nargs='+', help='Output columns to preserve')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--no_full', action='store_true', help='Run only training part')
    parser.add_argument('--full', action='store_true', help='Run training and prediction on test')
    
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument('--classification', action='store_true', help='Classification task')
    task_group.add_argument('--regression', action='store_true', help='Regression task')
    task_group.add_argument('--nlp', action='store_true', help='NLP task')
    
    parser.add_argument('--text_column', type=str, help='Text column for NLP tasks')
    parser.add_argument('--train_file', type=str, help='Specific train file name')
    parser.add_argument('--test_file', type=str, help='Specific test file name')
    parser.add_argument('--about', action='store_true', help='Show about information')
    parser.add_argument('--clear', action='store_true', help='Clear cache directories')
    
    return parser.parse_args()

def create_predictions_dataframe(pipeline, test_df, args):
    """Создать DataFrame с предсказаниями"""
    try:
        if hasattr(test_df, 'columns'):
            if args.nlp:
                X_test = test_df[args.text_column]
            else:
                X_test = test_df
            
            predictions = pipeline.predict(X_test)
            
            output_df = pd.DataFrame()
            
            id_col = None
            pred_col = 'prediction'
            
            if args.output_columns:
                for col in args.output_columns:
                    if col in test_df.columns:
                        id_col = col
                        output_df[col] = test_df[col].values
                        print(Fore.BLUE+f"Using {col} as ID column")
                        break
                
                if len(args.output_columns) > 1:
                    pred_col = args.output_columns[-1]
                    print(Fore.BLUE+f"Using '{pred_col}' as prediction column")
            
            if id_col is None:
                output_df['id'] = range(1, len(predictions) + 1)
                print(Fore.BLUE+"Generated sequential IDs")
            
            output_df[pred_col] = predictions
            
            return output_df, pred_col
        return None, None
    except Exception as e:
        print(Fore.RED+f"Error creating predictions: {e}")
        traceback.print_exc()
        return None, None

def main():
    show_logo()
    
    args = parse_arguments()
    
    if args.about:
        show_about()
        return
    
    if args.clear:
        clear_directories(['cache', 'results'])
        return
    
    if args.nlp and not args.text_column:
        print(Fore.RED+"Error: --text_column is required for NLP tasks")
        sys.exit(1)
    
    cache_dir = Path('cache')
    train_cache_dir = cache_dir / 'train'
    test_cache_dir = cache_dir / 'test'
    
    try:
        if args.path.endswith('.zip'):
            extract_path = cache_dir / 'extracted'
            extract_zip(args.path, extract_path)
            
            csv_files = find_csv_files(extract_path)
            
            if not csv_files:
                print(Fore.RED+"No CSV files found in archive")
                sys.exit(1)
            
            if args.train_file:
                train_file = extract_path / f"{args.train_file}.csv"
            else:
                train_keywords = ['train', 'training', 'learn']
                train_file = None
                for name, file in csv_files.items():
                    if any(keyword in name.lower() for keyword in train_keywords):
                        train_file = file
                        break
                
                if train_file is None:
                    train_file = list(csv_files.values())[0]
            
            if args.test_file:
                test_file = extract_path / f"{args.test_file}.csv"
            else:
                test_keywords = ['test', 'testing', 'eval']
                test_file = None
                for name, file in csv_files.items():
                    if any(keyword in name.lower() for keyword in test_keywords):
                        test_file = file
                        break
                
                if test_file is None and len(csv_files) > 1:
                    test_file = list(csv_files.values())[1]
                elif test_file is None:
                    test_file = train_file
            
            args.train_path = str(train_file)
            args.test_path = str(test_file)
            
            print(Fore.BLUE+f"Using train file: {train_file}")
            print(Fore.BLUE+f"Using test file: {test_file}")
        else:
            args.train_path = args.path
            args.test_path = args.path
        
        if args.classification:
            task_type = 'classification'
            preprocess_func = run_classifier_preprocessing
            pipeline_class = ClassifierPipeline
        elif args.regression:
            task_type = 'regression'
            preprocess_func = run_regressor_preprocessing
            pipeline_class = RegressorPipeline
        else:
            task_type = 'nlp'
            preprocess_func = run_nlp_preprocessing
            pipeline_class = NLPPipeline
        
        print(Fore.BLUE+f"\nStarting {task_type} with {args.algorithm}")
        print(Fore.BLUE+f"Target: {args.target}")
        print(Fore.BLUE+f"Output columns: {args.output_columns}")
        
        print(Fore.MAGENTA+"\nStep 1: Preprocessing training data...")
        args.path = args.train_path
        if not preprocess_func(args, is_training=True):
            print(Fore.RED+"Failed to preprocess training data")
            sys.exit(1)
        
        train_processed_path = train_cache_dir / f"{Path(args.train_path).stem}_processed.csv"
        if not train_processed_path.exists():
            print(Fore.RED+f"Processed train file not found: {train_processed_path}")
            sys.exit(1)
        
        train_df = pd.read_csv(train_processed_path)
        print(Fore.BLUE+f"Processed train data shape: {train_df.shape}")
        
        if args.target not in train_df.columns:
            print(Fore.RED+f"Target column '{args.target}' not found in processed data")
            sys.exit(1)
        
        if args.nlp:
            pipeline = pipeline_class(args.algorithm, args.target, task_type, args.save_model, args.output_columns)
            X_train = train_df[args.text_column]
            if args.target not in train_df.columns:
                print(Fore.RED+f"Target column '{args.target}' not found in processed data")
                sys.exit(1)
            y_train = train_df[args.target]
        else:
            pipeline = pipeline_class(args.algorithm, args.target, args.save_model, args.output_columns)
            X_train = train_df.drop(columns=[args.target])
            y_train = train_df[args.target]
        
        y_train = train_df[args.target]
        
        print(Fore.MAGENTA+"\nStep 2: Training model...")
        pipeline.train_model(X_train, y_train)
        
        if args.no_full:
            print(Fore.GREEN+"Training completed successfully!")
            return
        
        print(Fore.MAGENTA+"\nStep 3: Processing test data and making predictions...")
        
        args.path = args.test_path
        if not preprocess_func(args, is_training=False):
            print(Fore.RED+"Failed to preprocess test data")
            sys.exit(1)
        
        test_processed_path = test_cache_dir / f"{Path(args.test_path).stem}_processed.csv"
        if not test_processed_path.exists():
            print(Fore.RED+f"Processed test file not found: {test_processed_path}")
            sys.exit(1)
        
        test_df = pd.read_csv(test_processed_path)
        print(Fore.BLUE+f"Processed test data shape: {test_df.shape}")
        
        output_df, pred_col = create_predictions_dataframe(pipeline, test_df, args)
        
        if output_df is None:
            print(Fore.RED+"Failed to create predictions dataframe")
            sys.exit(1)
        
        predictions_path = Path('results') / 'predictions.csv'
        output_df.to_csv(predictions_path, index=False)
        print(Fore.GREEN+f"Predictions saved to: {predictions_path}")
        print(Fore.BLUE+f"Predictions shape: {output_df.shape}")
        print(Fore.BLUE+f"First 5 predictions:")
        print(output_df.head())
        
        if args.target in test_df.columns:
            print(Fore.MAGENTA+"\nStep 4: Evaluating model...")
            if args.nlp:
                X_test = test_df[args.text_column]
            else:
                X_test = test_df.drop(columns=[args.target])
            
            y_test = test_df[args.target]
            metrics = pipeline.evaluate(X_test, y_test)
            
            save_results(pipeline, output_df[pred_col].values, test_df, args, metrics, task_type)
        else:
            print(Fore.BLUE+"No target in test data, skipping evaluation")
            save_results(pipeline, output_df[pred_col].values, test_df, args, {}, task_type)
        
        print(Fore.GREEN+"\nPipeline completed successfully!")
        
    except Exception as e:
        print(Fore.RED+f"Error in pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()