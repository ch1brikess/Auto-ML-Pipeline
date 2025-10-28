# ML Pipeline - Automated Machine Learning Pipeline

## Overview

ML Pipeline is a comprehensive automated machine learning system designed for both classification and regression tasks. It provides end-to-end automation from data preprocessing to model training and prediction generation.

## Features

- **Automated Data Preprocessing**: Handles missing values, categorical encoding, feature scaling, and dimensionality reduction
- **Multiple Algorithms**: Supports popular ML algorithms for both classification and regression
- **Hyperparameter Tuning**: Automated hyperparameter optimization using GridSearchCV
- **Feature Alignment**: Ensures consistency between training and test datasets
- **Comprehensive Reporting**: Generates detailed reports, metrics, and prediction files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ch1brikess/Auto-ML-Pipeline.git
cd Auto-ML-Pipeline
```
2. Install dependencies
```bash
pip install -r req.txt
```
or
```bash
pip3 install -r req.txt
```

## Use

Example for use:
```bash
python main.py --path .\titanic.zip --target Transported --classification --algorithm DecisionTreeClassifier --output_columns PassengerId Transported
```

For more details use:
```bash
python main.py --help
```
