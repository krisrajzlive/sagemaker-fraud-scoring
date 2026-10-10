"""SageMaker-facing constants shared by local runs and AWS job definitions."""

# SageMaker scrapes training metrics from the container's stdout with these regexes (the
# ``metric_definitions`` of a training job). They match the line printed by
# ``fraud_scoring.training.train``: ``epoch=3 train_loss=0.01 val_pr_auc=0.8 val_roc_auc=0.97``.
METRIC_DEFINITIONS = [
    {"Name": "train:loss", "Regex": r"train_loss=([0-9.eE+-]+)"},
    {"Name": "validation:pr_auc", "Regex": r"val_pr_auc=([0-9.eE+-]+)"},
    {"Name": "validation:roc_auc", "Regex": r"val_roc_auc=([0-9.eE+-]+)"},
]

# Name of the metric that defines "better" when comparing runs or tuning.
OBJECTIVE_METRIC = "validation:pr_auc"
