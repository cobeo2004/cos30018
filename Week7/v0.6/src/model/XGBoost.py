from numpy import ndarray
from .IModel import IModel
from xgboost import XGBRegressor


class XGBoostModel(IModel):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        max_depth: int = 6,
        verbosity: int = 1,
    ) -> None:
        self.xgb_model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            verbosity=verbosity,
        )

    def train(self, x_train: ndarray, y_train: ndarray) -> XGBRegressor:
        self.xgb_model.fit(x_train, y_train)
        return self.xgb_model

    def predict(self, x_test: ndarray) -> ndarray:
        return self.xgb_model.predict(x_test)

    def score(self, x_test: ndarray, y_test: ndarray) -> float:
        return self.xgb_model.score(x_test, y_test)
