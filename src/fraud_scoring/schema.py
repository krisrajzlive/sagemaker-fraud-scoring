"""Column names shared by data preparation, feature engineering and the scoring API."""

TARGET = "Class"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]

# Fields a transaction must carry to be scored.
INPUT_FIELDS = ["Time", *V_COLUMNS, "Amount"]

# Columns of the raw dataset.
RAW_COLUMNS = [*INPUT_FIELDS, TARGET]

# Columns of the model input after feature engineering, in order.
FEATURE_NAMES = [*V_COLUMNS, "log_amount", "tod_sin", "tod_cos"]

SPLITS = ("train", "validation", "test")
