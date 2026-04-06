# MLOps Project: Streaming Binary Classification

MVP MLOps-система для обработки потоковых табличных данных на датасете Ethiopian Insurance Corporation.  
Проект реализует полный цикл работы с данными и моделями без специализированных MLOps-платформ: загрузку и хранение сырых данных, анализ качества, генерацию ассоциативных правил, обучение нескольких моделей, валидацию по времени, сериализацию артефактов, построение отчётов и inference на внешнем CSV-файле.

## Кратко о проекте

Система поддерживает:

- потоковую загрузку табличных данных через временные батчи;
- хранение сырых данных и результатов анализа в SQLite;
- data quality, ассоциативные правила и автоматический EDA;
- обучение `CatBoost` и `MLPClassifier`;
- time-based validation, сохранение артефактов и inference на внешнем CSV;
- CI-проверку локального обучающего сценария через GitHub Actions.

## Документация

Подробная документация вынесена в директорию [doc](doc).

- [doc/task.md](doc/task.md)  
  Постановка задачи и описание подхода.
- [doc/usage.md](doc/usage.md)  
  Развертывание, запуск и CLI.
- [doc/architecture.md](doc/architecture.md)  
  Архитектура пайплайна, структура проекта, БД, модели, inference и отчёты.
- [doc/ci.md](doc/ci.md)  
  Workflow второй части задания.
- [doc/grade.md](doc/grade.md)  
  Ожидаемые баллы по критериям оценки.

## Быстрый старт

```bash
source .venv/bin/activate
pip install -r requirements.txt

python run.py -mode update
python run.py -mode train
python run.py -mode summary
python run.py -mode inference -file "examples/inference_sample.csv"
```

Подробные инструкции по запуску, режимам CLI, конфигурации и зависимостям вынесены в [doc/usage.md](doc/usage.md).

## Развертывание

Для запуска на новой машине достаточно:

1. Установить Python `3.11`.
2. Создать виртуальное окружение.
3. Установить зависимости из [requirements.txt](requirements.txt).
4. Выполнить `python run.py -mode update`.
5. Выполнить `python run.py -mode train`.
6. Выполнить `python run.py -mode summary`.
7. Для прикладного использования запускать `python run.py -mode inference -file <input.csv>`.
