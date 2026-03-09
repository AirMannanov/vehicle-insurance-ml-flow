# MLOps Project: Streaming Binary Classification

MLOps pipeline MVP for binary classification on the Ethiopian Insurance Corporation dataset.

## Quick Start

Use the project virtualenv (e.g. `.venv`):

```bash
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
python run.py -mode update
python run.py -mode inference -file "path/to/data.csv"
python run.py -mode reset-db   # delete DB so next run re-seeds
```

Without activating: `.venv/bin/python run.py -mode update` (Linux/macOS).

## Project Structure

```
mlops-project/
├── run.py              # CLI entry point
├── config.yaml         # Pipeline configuration
├── src/                # Source code modules
├── data/               # Raw CSV data
├── artifacts/          # Serialized models
├── storage/            # SQLite database
├── reports/            # Generated reports
└── logs/               # Log files
```
