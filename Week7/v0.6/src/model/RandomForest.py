from numpy import ndarray
from .IModel import IModel
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel(IModel):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        min_samples_split: int = 2,
    ) -> None:
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

    def train(self, x_train: ndarray, y_train: ndarray) -> RandomForestRegressor:
        self.rf_model.fit(x_train, y_train)
        return self.rf_model

    def predict(self, x_test: ndarray) -> ndarray:
        return self.rf_model.predict(x_test)

    def score(self, x_test: ndarray, y_test: ndarray) -> float:
        return self.rf_model.score(x_test, y_test)
