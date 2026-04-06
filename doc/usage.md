# Usage And Deployment

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

## Развертывание

Минимальная схема развертывания проекта на новой машине:

1. Установить Python `3.11`.
2. Клонировать репозиторий.
3. Создать и активировать виртуальное окружение.
4. Установить зависимости из [requirements.txt](../requirements.txt).
5. Запустить `update`, затем `train`, затем `summary`.
6. Для прикладного использования запускать `inference` на внешнем CSV-файле.

За воспроизводимость окружения отвечает фиксация версий библиотек в [requirements.txt](../requirements.txt).

## CLI

Основная точка входа: [run.py](../run.py)

Поддерживаемые режимы:

- `update`  
  Загружает датасет, выполняет батчирование, сохраняет сырые данные в SQLite, считает Data Quality и association rules для новых батчей, затем переобучает модели и обновляет model reports.
- `bootstrap-local`  
  Загружает локальный CSV-файл в SQLite как временные батчи без внешней сети. Режим нужен для CI-friendly сценариев и локальной репетиции GitHub Actions.
- `train`  
  Обучает модели из `train_config.yaml`, валидирует их на временном разбиении, сохраняет артефакты и отчёты по запускам.
- `summary`  
  Генерирует сводный отчёт мониторинга в [reports/dq_report.md](../reports/dq_report.md).
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

Файл: [config.yaml](../config.yaml)

Здесь задаются:

- имя временной колонки и target;
- пороги очистки;
- настройки preprocessing;
- путь к SQLite;
- путь к summary report;
- настройки логирования.

### Конфиг обучения

Файл: [train_config.yaml](../train_config.yaml)

Здесь задаются:

- режим выбора датасета для обучения;
- train/validation/test split;
- primary metric;
- набор candidate thresholds;
- список включённых моделей;
- гиперпараметры `catboost` и `mlp`;
- директория для артефактов.

Для CI дополнительно используются:

- [config.ci.yaml](../config.ci.yaml)
- [train_config.ci.yaml](../train_config.ci.yaml)

Они настраивают отдельную SQLite-базу, отдельный лог-файл и облегчённые параметры обучения для GitHub Actions.

## Типовой сценарий работы

```bash
python run.py -mode update
python run.py -mode train
python run.py -mode summary
python run.py -mode inference -file examples/inference_sample.csv
```

После этого:

- база будет заполнена батчами и аналитикой;
- модели будут обучены и сериализованы;
- summary report будет лежать в `reports/dq_report.md`;
- inference создаст CSV с колонками `predict_proba` и `predict`.

## Зависимости

Список зависимостей находится в [requirements.txt](../requirements.txt).

Ключевые библиотеки:

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `catboost`
- `mlxtend`
- `matplotlib`
- `joblib`
- `PyYAML`
- `kagglehub`
