import pandas as pd
import numpy as np
import zipfile
import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from colorama import init, Fore

init(autoreset=True)

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
    while True:
        allow = input(Fore.GREEN+'Are you confirm it? [Y/n]: ')
        if allow.lower() in ['yes', 'y', '1', '']:
            allow = True
        else:
            allow = False
        
        if allow:
            for folder_path in paths:
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(Fore.RED+f'Error with delete file: {file_path}. {e}.')
            print(Fore.MAGENTA+'Done cleaning cache.')
            return
        else:
            print(Fore.MAGENTA+'Cancelled.')
            return

def save_results(pipeline, predictions, test_data, args, metrics, task_type):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    output_df = pd.DataFrame()
    
    passenger_id_col = None
    for col in args.output_columns:
        if col in test_data.columns:
            passenger_id_col = col
            break
    
    if passenger_id_col:
        output_df[passenger_id_col] = test_data[passenger_id_col].values
        print(Fore.BLUE+f"Using {passenger_id_col} as ID column with {len(test_data[passenger_id_col])} values")
    else:
        output_df['id'] = range(1, len(predictions) + 1)
        print(Fore.BLUE+"ID column not found, generated sequential IDs")
    
    if args.output_columns and len(args.output_columns) > 0:
        prediction_column = args.output_columns[-1]
        print(Fore.BLUE+f"Using '{prediction_column}' as prediction column name")
    else:
        prediction_column = 'Prediction'
        print(Fore.BLUE+f"Using default '{prediction_column}' as prediction column name")
    
    output_df[prediction_column] = predictions
    
    for col in args.output_columns:
        if (col in test_data.columns and 
            col != passenger_id_col and 
            col != prediction_column):
            output_df[col] = test_data[col].values
            print(Fore.BLUE+f"Added output column: {col}")
    
    predictions_path = results_dir / 'predictions.csv'
    output_df.to_csv(predictions_path, index=False)
    
    results_info = {
        'algorithm': args.algorithm,
        'task_type': task_type,
        'target_column': args.target,
        'best_parameters': pipeline.best_params,
        'metrics': metrics,
        'output_columns': args.output_columns,
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
    
    print(Fore.MAGENTA+f"\nResults saved to {results_dir}:")
    print(Fore.BLUE+f"   - Predictions: {predictions_path}")
    print(Fore.BLUE+f"   - Training info: {info_path}")
    print(Fore.BLUE+f"   - Report: {report_path}")
    print(Fore.BLUE+f"   - Output shape: {output_df.shape}")
    
    if len(output_df) == 0:
        print(Fore.RED+"ERROR: Output file is empty!")
    else:
        print(Fore.GREEN+f"SUCCESS: Generated {len(output_df)} predictions")

def show_logo():
    print(Fore.RED+'================================')
    logo = """
        ╔║║╗╦╗╔╗╦║╔╔═╔╗╔╗
        ║╠╣║╠╣╠╝║╠╣╠═╚╗╚╗
        ╚║║╩╩╝╠╗╩║╚╚═╚╝╚╝
"""
    print(Fore.RED+logo)
    print(f" {Fore.BLUE+'Parsing you...'} {Fore.RED+'|'} {Fore.MAGENTA+'by ch1brikess'}")
    print(Fore.RED+'================================')

def show_about():
    about_file = Path(__file__).parent.parent.parent / 'about.txt'
    if about_file.exists():
        with open(about_file, 'r', encoding='utf-8') as f:
            about_text = f.read()
        print(Fore.GREEN+about_text)
    else:
        print(Fore.BLUE+"ML Pipeline - Automated Machine Learning Pipeline")
        print(Fore.MAGENTA+"Version: 2.0")
        print(Fore.BLUE+"Description: Modular ML pipeline for classification, regression and NLP tasks")