# ML Pipeline - Automated Machine Learning Pipeline
![95403395-42a0-40fc-83e5-2ec38b885159](https://github.com/user-attachments/assets/133c77d8-8684-4ca9-8f2f-7d538bdbb180)

# Languages
1. [EN](#EN)
2. [RU](#RU)

# EN

## Overview

ML Pipeline is a comprehensive automated machine learning system designed for both classification and regression tasks. It provides end-to-end automation from data preprocessing to model training and prediction generation.

## Key Features

- **Automated Data Preprocessing**: Handles missing values, categorical encoding, feature scaling, and dimensionality reduction
- **Multiple Algorithms**: Supports popular ML algorithms for both classification and regression
- **Hyperparameter Tuning**: Automated hyperparameter optimization using GridSearchCV
- **Feature Alignment**: Ensures consistency between training and test datasets
- **Comprehensive Reporting**: Generates detailed reports, metrics, and prediction files

## Supported Algorithms

### Classification Algorithms
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Gradient Boosting Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors Classifier

### Regression Algorithms
- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ch1brikess/-Auto-ML-Pipeline.git
cd -Auto-ML-Pipeline
```

2. Install required dependencies:
```bash
pip install -r req.txt
```
or
```bash
pip3 install -r req.txt
```

## Usage Methods

### Method 1: Using Graphical Interface (Recommended for Beginners)

#### Option A: Run Python script directly
```bash
python ./gui-qt5.py
```
or
```bash
python ./gui-qt6.py
```

#### Option B: Build executable file
```bash
pyinstaller MLPipline-qt5.spec
```
or
```bash
pyinstaller MLPipline-qt6.spec
```

### Method 2: Using Command Line (For Advanced Users)

#### Basic classification example:
```bash
python main.py --path ./titanic.zip --target Transported --classification --algorithm DecisionTreeClassifier --output_columns PassengerId Transported
```

#### Basic regression example:
```bash
python main.py --path ./housing_data.zip --target price --regression --algorithm RandomForestRegressor --output_columns id price
```

#### Get full help:
```bash
python main.py --help
```

## Step-by-Step Guide for Beginners

### Using the Graphical Interface:

1. **Launch the application** by running `gui-qt5.py` or the executable file
2. **Load your data** - Click "Browse" and select your dataset file (CSV, Excel, or ZIP)
3. **Configure settings**:
   - Select target variable (what you want to predict)
   - Choose problem type: Classification or Regression
   - Select machine learning algorithm
   - Specify which columns to include in output
4. **Run the pipeline** - Click "Run ML Pipeline"
5. **Get results** - The system will generate predictions and detailed reports

### Using Command Line:

1. **Prepare your data** in CSV format
2. **Identify your target variable** (the column you want to predict)
3. **Choose the right algorithm** based on your problem type
4. **Run the command** with appropriate parameters
5. **Check the output files** for predictions and analysis

## Input Data Requirements

- Supported formats: CSV, Excel, ZIP containing data files
- Data should be in tabular format with columns as features
- First row should contain column names
- Missing values are handled automatically

## Output Files

- `predictions.csv` - Final predictions with specified output columns
- `metrics.txt` - Detailed performance metrics and model information
- `preprocessing_info.txt` - Information about data preprocessing steps
- `feature_importance.png` - Visualization of important features (if supported by algorithm)

## Common Use Cases

- **Customer Churn Prediction** (Classification)
- **House Price Prediction** (Regression)
- **Spam Detection** (Classification)
- **Sales Forecasting** (Regression)
- **Medical Diagnosis** (Classification)

## Troubleshooting

- If installation fails, ensure you have Python 3.7+ installed
- For large datasets, processing may take several minutes
- Ensure your target variable column exists in the dataset
- Check that output columns are spelled correctly

# RU

## Обзор

ML Pipeline — это комплексная автоматизированная система машинного обучения, предназначенная для задач классификации и регрессии. Она обеспечивает сквозную автоматизацию от предварительной обработки данных до обучения модели и генерации прогнозов.

## Основные возможности

- **Автоматизированная предварительная обработка данных**: обработка пропущенных значений, категориальное кодирование, масштабирование признаков и снижение размерности
- **Поддержка нескольких алгоритмов**: популярные алгоритмы машинного обучения для классификации и регрессии
- **Настройка гиперпараметров**: автоматическая оптимизация гиперпараметров с помощью GridSearchCV
- **Выравнивание признаков**: обеспечение согласованности между обучающими и тестовыми наборами данных
- **Подробная отчётность**: создание подробных отчётов, метрик и файлов прогнозов

## Поддерживаемые алгоритмы

### Алгоритмы классификации
- Логистическая регрессия
- Случайный лес (классификация)
- Дерево решений (классификация)
- Градиентный бустинг (классификация)
- Метод опорных векторов (SVC)
- Метод k-ближайших соседей (классификация)

### Алгоритмы регрессии
- Линейная регрессия
- Случайный лес (регрессия)
- Дерево решений (регрессия)
- Градиентный бустинг (регрессия)
- Метод опорных векторов (SVR)
- Метод k-ближайших соседей (регрессия)

## Установка

1. Скопируйте репозиторий:
```bash
git clone https://github.com/ch1brikess/-Auto-ML-Pipeline.git
cd -Auto-ML-Pipeline
```

2. Установите необходимые зависимости:
```bash
pip install -r req.txt
```
или
```bash
pip3 install -r req.txt
```

## Способы использования

### Способ 1: Использование графического интерфейса (Рекомендуется для начинающих)

#### Вариант A: Запуск Python скрипта напрямую
```bash
python ./gui-qt5.py
```
или
```bash
python ./gui-qt6.py
```

#### Вариант B: Сборка исполняемого файла
```bash
pyinstaller MLPipline-qt5.spec
```
или
```bash
pyinstaller MLPipline-qt6.spec
```

### Способ 2: Использование командной строки (Для продвинутых пользователей)

#### Базовый пример классификации:
```bash
python main.py --path ./titanic.zip --target Transported --classification --algorithm DecisionTreeClassifier --output_columns PassengerId Transported
```

#### Базовый пример регрессии:
```bash
python main.py --path ./housing_data.zip --target price --regression --algorithm RandomForestRegressor --output_columns id price
```

#### Получить полную справку:
```bash
python main.py --help
```

## Пошаговое руководство для начинающих

### Использование графического интерфейса:

1. **Запустите приложение** выполнив `gui-qt5.py` или через исполняемый файл
2. **Загрузите ваши данные** - Нажмите "Browse" и выберите файл с данными (CSV, Excel или ZIP)
3. **Настройте параметры**:
   - Выберите целевую переменную (то, что хотите предсказать)
   - Выберите тип задачи: Классификация или Регрессия
   - Выберите алгоритм машинного обучения
   - Укажите, какие столбцы включить в результат
4. **Запустите pipeline** - Нажмите "Run ML Pipeline"
5. **Получите результаты** - Система сгенерирует прогнозы и подробные отчёты

### Использование командной строки:

1. **Подготовьте данные** в формате CSV
2. **Определите целевую переменную** (столбец, который хотите предсказать)
3. **Выберите подходящий алгоритм** в зависимости от типа задачи
4. **Запустите команду** с соответствующими параметрами
5. **Проверьте выходные файлы** на наличие прогнозов и анализа

## Требования к входным данным

- Поддерживаемые форматы: CSV, Excel, ZIP с файлами данных
- Данные должны быть в табличном формате со столбцами как признаки
- Первая строка должна содержать названия столбцов
- Пропущенные значения обрабатываются автоматически

## Выходные файлы

- `predictions.csv` - Финальные прогнозы с указанными выходными столбцами
- `metrics.txt` - Детальные метрики производительности и информация о модели
- `preprocessing_info.txt` - Информация о шагах предварительной обработки данных
- `feature_importance.png` - Визуализация важных признаков (если поддерживается алгоритмом)

## Типичные сценарии использования

- **Прогнозирование оттока клиентов** (Классификация)
- **Предсказание цен на недвижимость** (Регрессия)
- **Обнаружение спама** (Классификация)
- **Прогнозирование продаж** (Регрессия)
- **Медицинская диагностика** (Классификация)

## Решение проблем

- Если установка не удалась, убедитесь что у вас установлен Python 3.7+
- Для больших наборов данных обработка может занять несколько минут
- Убедитесь что столбец целевой переменной существует в наборе данных
- Проверьте правильность написания выходных столбцов
