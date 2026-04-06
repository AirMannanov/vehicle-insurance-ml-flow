# Architecture

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
   Система скачивает исходные CSV, объединяет их, разбивает данные на месячные батчи и записывает их в SQLite.
2. `update`  
   Для новых батчей считаются Data Quality-метрики и ассоциативные правила, после чего модели переобучаются на актуальном наборе данных.
3. `train`  
   Из БД извлекаются строки, строится временной split, обучаются модели, считаются validation/test metrics, сохраняются артефакты и markdown-отчёты.
4. `summary`  
   Формируется итоговый отчёт по качеству данных и автоматический EDA.
5. `inference`  
   Загружается лучшая отмеченная модель, применяется к внешнему CSV и сохраняется файл с предсказаниями.

## Структура проекта

### Корень репозитория

- [run.py](../run.py)  
  CLI entrypoint и маршрутизация режимов работы.
- [config.yaml](../config.yaml)  
  Общая конфигурация пайплайна.
- [train_config.yaml](../train_config.yaml)  
  Конфигурация обучения и валидации.
- [requirements.txt](../requirements.txt)  
  Python-зависимости проекта.
- [examples/inference_sample.csv](../examples/inference_sample.csv)  
  Пример входного файла для inference.
- [examples/ci_training_sample.csv](../examples/ci_training_sample.csv)  
  Компактный локальный sample-датасет для CI и локальной репетиции workflow.

### Исходный код

- [src/data](../src/data)  
  Загрузка датасета, чтение CSV, генерация батчей, сохранение и чтение сырых данных.
- [src/analysis](../src/analysis)  
  Data Quality, association rules, очистка и генерация summary report.
- [src/preprocessing](../src/preprocessing)  
  Подготовка данных для `MLP` и `CatBoost`, обработка пропусков, категориальных и числовых признаков.
- [src/models](../src/models)  
  Обучение моделей и выбор датасета для train.
- [src/validation](../src/validation)  
  Time-based split, подбор порога и расчёт метрик.
- [src/database](../src/database)  
  Подключение к SQLite, миграции, row loaders, запросы по времени, reset.
- [src/serving](../src/serving)  
  Сохранение и загрузка bundle-артефактов моделей.
- [src/reporting](../src/reporting)  
  Генерация markdown-отчётов по validation runs.
- [src/tools](../src/tools)  
  Работа с конфигом и логированием.

### Данные и артефакты

- [storage](../storage)  
  SQLite-база проекта.
- [artifacts](../artifacts)  
  Сериализованные модели в формате `joblib`.
- [reports](../reports)  
  Summary report, figures и markdown-отчёты по моделям.
- [logs](../logs)  
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

- validation metrics;
- test metrics;
- собственный serialized bundle;
- markdown-отчёт с графиком;
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

- [examples/inference_sample.csv](../examples/inference_sample.csv)

Пример результата:

- [examples/inference_sample_predictions.csv](../examples/inference_sample_predictions.csv)

## Отчёты

### Summary report

Файл: [reports/dq_report.md](../reports/dq_report.md)

Содержит:

- агрегаты по батчам;
- агрегаты по признакам;
- график динамики missing rate / unique count;
- автоматический EDA;
- числовые распределения;
- корреляционную матрицу;
- top values по категориальным признакам.

### Model validation reports

Файлы: [reports/models](../reports/models)

Содержат:

- метрики validation/test;
- confusion matrix;
- class balance;
- split summary;
- информацию о feature space;
- гиперпараметры;
- график сравнения метрик.
