import pandas as pd
from .env import env
from fredapi import Fred
from src.processing import start, end


class FredHelper:
    def __init__(self) -> None:
        self.fred = Fred(api_key=env.api_key)

    def get_macroeconomic_data(
        self, start_date: str = start, end_date: str = end
    ) -> pd.DataFrame:

        gdp = self.fred.get_series("GDP", start_date=start_date, end_date=end_date)
        unemployment = self.fred.get_series(
            "UNRATE", start_date=start_date, end_date=end_date
        )
        inflation = self.fred.get_series(
            "CPIAUCSL", start_date=start_date, end_date=end_date
        )

        return pd.DataFrame(
            {
                "GDP": gdp,
                "Unemployment": unemployment,
                "Inflation": inflation,
            }
        )

