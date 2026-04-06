# CI And Workflow

## CI/CD

Базовый workflow для второй части задания находится в [.github/workflows/ci.yml](../.github/workflows/ci.yml).

Он автоматически запускается на:

- `push`
- `pull_request`

## Что делает workflow

1. Выполняет checkout репозитория.
2. Устанавливает Python `3.11`.
3. Устанавливает зависимости из `requirements.txt`.
4. Выполняет `reset` для очистки CI-артефактов.
5. Загружает локальный sample CSV в SQLite через `bootstrap-local`.
6. Обучает `CatBoost` и `MLP` на sample-данных.
7. Выгружает лог обучения как GitHub artifact `training-logs`.

## Зачем выбран такой подход

Для CI используется локальный sample-датасет и отдельные конфиги `config.ci.yaml` и `train_config.ci.yaml`.

Это даёт следующие преимущества:

- workflow не зависит от внешних источников данных во время проверки;
- запуск остаётся быстрым и повторяемым;
- обучающий pipeline проверяется end-to-end;
- логи сохраняются как артефакты и доступны после завершения job.

## Локальная репетиция workflow

```bash
python run.py -mode reset -config config.ci.yaml -train-config train_config.ci.yaml
python run.py -mode bootstrap-local -config config.ci.yaml -file examples/ci_training_sample.csv
python run.py -mode train -config config.ci.yaml -train-config train_config.ci.yaml
```
