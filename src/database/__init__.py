from src.database.connection import Database
from src.database.dataset_schema import FeatureType, column_in_schema, get_feature_type
from src.database.migrator import Migrator
from src.database.model_registry import (
    ModelRegistryRecord,
    insert_model_registry_record,
    list_model_registry_records,
    list_model_registry_records_by_name,
)
from src.database.model_validation_runs import (
    ModelValidationRunRecord,
    clear_selected_model_for_name,
    get_selected_model_validation_run,
    insert_model_validation_run,
    list_model_validation_runs,
    mark_model_validation_run_selected,
    update_model_validation_run_test_metrics,
)
from src.database.row_loader import iter_ml_batches, load_rows_by_ids, load_split_dataframe
from src.database.reset import reset_db, reset_project_outputs
from src.database.time_queries import (
    get_all_row_ids,
    get_available_event_dates_for_date_range,
    get_available_event_dates_for_exact_dates,
    get_event_date_counts,
    get_row_ids_for_date_range,
    get_row_ids_for_exact_dates,
    get_row_ids_for_source_batches,
    list_available_event_dates,
    list_available_event_times,
)

__all__ = [
    "FeatureType",
    "column_in_schema",
    "Database",
    "get_all_row_ids",
    "get_available_event_dates_for_date_range",
    "get_available_event_dates_for_exact_dates",
    "get_event_date_counts",
    "get_feature_type",
    "get_row_ids_for_date_range",
    "get_row_ids_for_exact_dates",
    "get_row_ids_for_source_batches",
    "iter_ml_batches",
    "insert_model_registry_record",
    "insert_model_validation_run",
    "list_available_event_dates",
    "list_available_event_times",
    "list_model_registry_records",
    "list_model_registry_records_by_name",
    "list_model_validation_runs",
    "load_rows_by_ids",
    "load_split_dataframe",
    "ModelValidationRunRecord",
    "ModelRegistryRecord",
    "Migrator",
    "mark_model_validation_run_selected",
    "reset_db",
    "reset_project_outputs",
    "clear_selected_model_for_name",
    "get_selected_model_validation_run",
    "update_model_validation_run_test_metrics",
]
