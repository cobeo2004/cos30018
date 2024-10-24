from .IModel import IModel
from .XGBoost import XGBoostModel
from .RandomForest import RandomForestModel

XGBoostInstance = XGBoostModel()
RandomForestInstance = RandomForestModel()

__all__ = ["IModel", "XGBoostModel", "RandomForestModel", "XGBoostInstance", "RandomForestInstance"]
