from typing_extensions import Annotated, Doc
import pandas as pd
import matplotlib.pyplot as plt
from src.processing.constants import ticker


def draw_box_chart(df: Annotated[pd.DataFrame, Doc("The data to be plotted")],
                   n: Annotated[int | None, Doc(
                       "Resampling period in trading days for aggregation, default is None")] = None,
                   k: Annotated[int, Doc(
                       "The interval of the box plot, default is 10")] = 10
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

    chart_data: list = []
    labels: list = []

    for index, row in cp_df.iterrows():
        chart_data.append([row['Close'], row['Open'], row['Low'], row['High']])
        labels.append(index.strftime('%Y-%m-%d'))

    figure, axes = plt.subplots()
    axes.boxplot(chart_data, vert=True, patch_artist=True)
    axes.set_xticklabels(labels)
    axes.set_title(f'{ticker} Box Plot Chart')
    axes.set_xlabel('Date')
    axes.set_ylabel('Price')
    axes.set_xticks(range(1, len(labels) + 1, k))
    axes.set_xticklabels(labels[::k], rotation=90)

    plt.show()
