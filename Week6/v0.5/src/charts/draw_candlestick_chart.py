from typing_extensions import Annotated, Doc
import pandas as pd
import mplfinance as mpf
from src.processing.constants import ticker


def draw_candlestick_chart(df: Annotated[pd.DataFrame, Doc("The data to be plotted")],
                           n: Annotated[int | None, Doc(
                               "Resampling period in trading days for aggregation, default is None")] = None,
                           ):
    cp_df = df.copy()
    if n is not None:
        cp_df = cp_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    cp_df['MA25'] = cp_df['Close'].rolling(window=25).mean()
    cp_df['MA100'] = cp_df['Close'].rolling(window=100).mean()
    cp_df['MA200'] = cp_df['Close'].rolling(window=200).mean()

    sub_chart: mpf.make_addplot = []
    if cp_df.MA25.dropna().shape[0] > 0:
        fixed_MA25 = cp_df.MA25.dropna().reindex(cp_df.index, fill_value=None)
        sub_chart.append(mpf.make_addplot(fixed_MA25, color='blue'))
    if cp_df.MA100.dropna().shape[0] > 0:
        fixed_MA100 = cp_df.MA100.dropna().reindex(cp_df.index, fill_value=None)
        sub_chart.append(mpf.make_addplot(fixed_MA100, color='orange'))
    if cp_df.MA200.dropna().shape[0] > 0:
        fixed_MA200 = cp_df.MA200.dropna().reindex(cp_df.index, fill_value=None)
        sub_chart.append(mpf.make_addplot(fixed_MA200, color='green'))

    mpf.plot(cp_df, type='candle', style='charles', ylabel='Price', ylabel_lower='Volume',
             volume=True, addplot=sub_chart, title=f'{ticker} Stock Price with Moving Averages')
