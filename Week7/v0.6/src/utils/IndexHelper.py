from typing import Tuple
import pandas as pd

class IndexHelper:
    def calculateRSI(self, data: pd.DataFrame, period: int = 21, targetColumn: str = "Close") -> pd.Series:
        delta = data[targetColumn].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculateMACD(
        self,
        data: pd.DataFrame,
        shortEMA: int = 12,
        longEMA: int = 26,
        signalSMA: int = 9,
        targetColumn: str = "Close",
    ) -> Tuple[pd.Series, pd.Series]:
        shortEMA = data[targetColumn].ewm(span=shortEMA, adjust=False).mean()
        longEMA = data[targetColumn].ewm(span=longEMA, adjust=False).mean()
        macd = shortEMA - longEMA
        signal = macd.ewm(span=signalSMA, adjust=False).mean()
        return macd, signal
