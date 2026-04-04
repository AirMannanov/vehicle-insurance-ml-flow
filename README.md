# MLOps Project: Streaming Binary Classification

MVP MLOps-система для обработки потоковых табличных данных на датасете Ethiopian Insurance Corporation.  
Проект реализует полный цикл работы с данными и моделями без специализированных MLOps-платформ: загрузку и хранение сырых данных, анализ качества, генерацию ассоциативных правил, обучение нескольких моделей, валидацию по времени, сериализацию артефактов, построение отчётов и inference на внешнем CSV-файле.

## Что умеет программа

- Загружать исходный датасет и эмулировать поток данных через временные батчи.
- Хранить сырые данные, метаинформацию и результаты анализа в SQLite.
- Считать показатели Data Quality по каждому батчу и признаку.
- Строить ассоциативные правила на бинаризованных признаках.
- Выполнять базовую очистку и предобработку признаков.
- Обучать две модели: `CatBoost` и `MLPClassifier`.
- Делать time-based split на train/validation/test.
- Подбирать рабочий порог классификации по целевой метрике.
- Сохранять артефакты моделей и отчёты по валидационным прогонам.
- Генерировать сводный monitoring report.
- Применять лучшую выбранную модель к внешнему CSV-файлу.

## Архитектура пайплайна

Общий поток работы:

1. `update`
   Система скачивает исходные CSV, объединяет их, разбивает на месячные батчи и записывает данные в SQLite.
2. `update`
   Для новых батчей считаются Data Quality-метрики и ассоциативные правила, после чего модели переобучаются на актуальном наборе данных.
3. `train`
   Из БД извлекаются строки, строится временной split, обучаются модели, считаются validation/test metrics, сохраняются артефакты и markdown-отчёты. Этот режим нужен, если нужно переобучить модели без повторной загрузки данных.
4. `summary`
   Формируется итоговый отчёт по качеству данных и автоматический EDA.
5. `inference`
   Загружается лучшая отмеченная модель, применяется к внешнему CSV и сохраняется файл с предсказаниями.

## Быстрый старт

Используйте виртуальное окружение проекта, например `.venv`.

```bash
source .venv/bin/activate
pip install -r requirements.txt

python run.py -mode update
python run.py -mode train
python run.py -mode summary
python run.py -mode inference -file "examples/inference_sample.csv"
```

Без активации окружения:

```bash
.venv/bin/python run.py -mode update
```

Сброс всех сгенерированных артефактов:

```bash
python run.py -mode reset
```

## CLI

Основная точка входа: [run.py](/Users/mannanovairat/codebase/MSU/mlops-project/run.py)

Поддерживаемые режимы:

- `update`  
  Загружает датасет, выполняет батчирование, сохраняет сырые данные в SQLite, считает Data Quality и association rules для новых батчей, затем переобучает модели и обновляет model reports.
- `bootstrap-local`  
  Загружает локальный CSV-файл в SQLite как временные батчи без внешней сети. Режим нужен для CI-friendly сценариев и локальной репетиции GitHub Actions.
- `train`  
  Обучает модели из `train_config.yaml`, валидирует их на временном разбиении, сохраняет артефакты и отчёты по запускам.
- `summary`  
  Генерирует сводный отчёт мониторинга в [reports/dq_report.md](/Users/mannanovairat/codebase/MSU/mlops-project/reports/dq_report.md).
- `inference`  
  Применяет лучшую выбранную модель к внешнему CSV и сохраняет рядом файл `<input_name>_predictions.csv`.
- `reset`  
  Удаляет БД, логи, отчёты и артефакты моделей.

Пример:

```bash
python run.py -mode inference -file "examples/inference_sample.csv"
```

Для локального bootstrap:

```bash
python run.py -mode bootstrap-local -config config.ci.yaml -file "examples/ci_training_sample.csv"
```

## Конфигурация

### Основной конфиг

Файл: [config.yaml](/Users/mannanovairat/codebase/MSU/mlops-project/config.yaml)

Здесь задаются:

- имя временной колонки и target,
- пороги очистки,
- настройки preprocessing,
- путь к SQLite,
- путь к summary report,
- настройки логирования.

### Конфиг обучения

Файл: [train_config.yaml](/Users/mannanovairat/codebase/MSU/mlops-project/train_config.yaml)

Здесь задаются:

- режим выбора датасета для обучения,
- train/validation/test split,
- primary metric,
- набор candidate thresholds,
- список включённых моделей,
- гиперпараметры `catboost` и `mlp`,
- директория для артефактов.

Для CI дополнительно используются:

- [config.ci.yaml](/Users/mannanovairat/codebase/MSU/mlops-project/config.ci.yaml)
- [train_config.ci.yaml](/Users/mannanovairat/codebase/MSU/mlops-project/train_config.ci.yaml)

Они настраивают отдельную SQLite-базу, отдельный лог-файл и облегчённые параметры обучения для GitHub Actions.

## Структура проекта

### Корень репозитория

- [run.py](/Users/mannanovairat/codebase/MSU/mlops-project/run.py)  
  CLI entrypoint и маршрутизация режимов работы.
- [config.yaml](/Users/mannanovairat/codebase/MSU/mlops-project/config.yaml)  
  Общая конфигурация пайплайна.
- [train_config.yaml](/Users/mannanovairat/codebase/MSU/mlops-project/train_config.yaml)  
  Конфигурация обучения и валидации.
- [requirements.txt](/Users/mannanovairat/codebase/MSU/mlops-project/requirements.txt)  
  Python-зависимости проекта.
- [examples/inference_sample.csv](/Users/mannanovairat/codebase/MSU/mlops-project/examples/inference_sample.csv)  
  Пример входного файла для inference.
- [examples/ci_training_sample.csv](/Users/mannanovairat/codebase/MSU/mlops-project/examples/ci_training_sample.csv)  
  Компактный локальный sample-датасет для CI и локальной репетиции workflow.

### Исходный код

- [src/data](/Users/mannanovairat/codebase/MSU/mlops-project/src/data)  
  Загрузка датасета, чтение CSV, генерация батчей, сохранение и чтение сырых данных.
- [src/analysis](/Users/mannanovairat/codebase/MSU/mlops-project/src/analysis)  
  Data Quality, association rules, очистка и генерация summary report.
- [src/preprocessing](/Users/mannanovairat/codebase/MSU/mlops-project/src/preprocessing)  
  Подготовка данных для `MLP` и `CatBoost`, обработка пропусков, категориальных и числовых признаков.
- [src/models](/Users/mannanovairat/codebase/MSU/mlops-project/src/models)  
  Обучение моделей и выбор датасета для train.
- [src/validation](/Users/mannanovairat/codebase/MSU/mlops-project/src/validation)  
  Time-based split, подбор порога и расчёт метрик.
- [src/database](/Users/mannanovairat/codebase/MSU/mlops-project/src/database)  
  Подключение к SQLite, миграции, row loaders, запросы по времени, reset.
- [src/serving](/Users/mannanovairat/codebase/MSU/mlops-project/src/serving)  
  Сохранение и загрузка bundle-артефактов моделей.
- [src/reporting](/Users/mannanovairat/codebase/MSU/mlops-project/src/reporting)  
  Генерация markdown-отчётов по validation runs.
- [src/tools](/Users/mannanovairat/codebase/MSU/mlops-project/src/tools)  
  Работа с конфигом и логированием.

### Данные и артефакты

- [storage](/Users/mannanovairat/codebase/MSU/mlops-project/storage)  
  SQLite-база проекта.
- [artifacts](/Users/mannanovairat/codebase/MSU/mlops-project/artifacts)  
  Сериализованные модели в формате `joblib`.
- [reports](/Users/mannanovairat/codebase/MSU/mlops-project/reports)  
  Summary report, figures и markdown-отчёты по моделям.
- [logs](/Users/mannanovairat/codebase/MSU/mlops-project/logs)  
  Логи выполнения пайплайна.

## Что сохраняется в БД

SQLite используется как единое хранилище проекта.

Основные таблицы:

- `raw_batches`  
  Метаданные батчей.
- `raw_data`  
  Сырые строки датасета в JSON-представлении и временные поля `event_time` / `event_date`.
- `data_quality`  
  Показатели качества данных по батчам и признакам.
- `assoc_rules`  
  Ассоциативные правила для батчей.
- `model_validation_runs`  
  История запусков обучения, метрики, конфиги split и флаг выбранной модели.

## Модели

Поддерживаются две модели:

- `catboost`  
  Градиентный бустинг на деревьях, устойчивый к категориальным признакам.
- `mlp`  
  Нейронная сеть на базе `sklearn.neural_network.MLPClassifier`.

После обучения каждая модель получает:

- validation metrics,
- test metrics,
- собственный serialized bundle,
- markdown-отчёт с графиком,
- запись в `model_validation_runs`.

## Inference

Inference работает по следующей схеме:

1. Из БД выбираются модели, отмеченные как `selected`.
2. Из них берётся лучшая по `validation f1`.
3. Загружается соответствующий `joblib` bundle.
4. Входной CSV приводится к нужному feature space.
5. Считаются `predict_proba` и `predict`.
6. Результат сохраняется рядом с исходным файлом.

Пример входного файла:

- [examples/inference_sample.csv](/Users/mannanovairat/codebase/MSU/mlops-project/examples/inference_sample.csv)

Пример результата:

- [examples/inference_sample_predictions.csv](/Users/mannanovairat/codebase/MSU/mlops-project/examples/inference_sample_predictions.csv)

## Отчёты

### Summary report

Файл: [reports/dq_report.md](/Users/mannanovairat/codebase/MSU/mlops-project/reports/dq_report.md)

Содержит:

- агрегаты по батчам,
- агрегаты по признакам,
- график динамики missing rate / unique count,
- автоматический EDA,
- числовые распределения,
- корреляционную матрицу,
- top values по категориальным признакам.

### Model validation reports

Файлы: [reports/models](/Users/mannanovairat/codebase/MSU/mlops-project/reports/models)

Содержат:

- метрики validation/test,
- confusion matrix,
- class balance,
- split summary,
- информацию о feature space,
- гиперпараметры,
- график сравнения метрик.

## Типовой сценарий работы

```bash
python run.py -mode update
python run.py -mode train
python run.py -mode summary
python run.py -mode inference -file examples/inference_sample.csv
```

После этого:

- база будет заполнена батчами и аналитикой,
- модели будут обучены и сериализованы,
- summary report будет лежать в `reports/dq_report.md`,
- inference создаст CSV с колонками `predict_proba` и `predict`.

## CI/CD

Базовый workflow для второй задачи находится в [.github/workflows/ci.yml](/Users/mannanovairat/codebase/MSU/mlops-project/.github/workflows/ci.yml).

Он автоматически запускается на:

- `push`
- `pull_request`

Сценарий workflow:

1. checkout репозитория;
2. установка Python 3.11;
3. установка зависимостей из `requirements.txt`;
4. сброс CI-артефактов через `reset`;
5. загрузка локального sample CSV в SQLite через `bootstrap-local`;
6. обучение `CatBoost` и `MLP` на sample-данных;
7. выгрузка логов как GitHub artifact `training-logs`.

Локальная репетиция тех же шагов:

```bash
python run.py -mode reset -config config.ci.yaml -train-config train_config.ci.yaml
python run.py -mode bootstrap-local -config config.ci.yaml -file examples/ci_training_sample.csv
python run.py -mode train -config config.ci.yaml -train-config train_config.ci.yaml
```

## Зависимости

Список зависимостей находится в [requirements.txt](/Users/mannanovairat/codebase/MSU/mlops-project/requirements.txt).

Ключевые библиотеки:

- `pandas`
- `scikit-learn`
- `catboost`
- `mlxtend`
- `matplotlib`
- `joblib`
- `pyyaml`
- `kagglehub`
