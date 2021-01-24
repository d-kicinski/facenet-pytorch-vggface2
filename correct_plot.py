from dataclasses import dataclass
from typing import List

import csv
from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class TimeSeriesData:
    x: List[int]
    y: List[float]


def read_data_file(path: Path):
    data = TimeSeriesData([], [])
    with path.open("r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            data.x.append(int(row[1]))
            data.y.append(float(row[2]))
    return data


def plot_time_series_data(data: TimeSeriesData):
    plt.plot(data.x, data.y)
    plt.show()


def multiply(data1: TimeSeriesData, data2: TimeSeriesData) -> TimeSeriesData:
    data = TimeSeriesData(data1.x, [])
    for d1, d2 in zip(data1.y, data2.y):
        data.y.append(d1 * d2)
    return data


def main():
    data_loss: TimeSeriesData = read_data_file(Path("run-.-tag-loss_train.csv"))
    data_iters: TimeSeriesData = read_data_file(Path("run-.-tag-iters_to_accumulate.csv"))
    plot_time_series_data(data_iters)
    plot_time_series_data(data_loss)

    data: TimeSeriesData = multiply(data_loss, data_iters)
    plot_time_series_data(data)


if __name__ == '__main__':
    main()
