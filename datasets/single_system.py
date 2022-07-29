import os
import pathlib
from random import Random
from pathlib import Path

import numpy
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor

from analysis import HolisticAnalyis
from assignment import HOPAssignment, walk_random_priorities, walk_random_priorities_processors
from data import write_to_csv, get_xy, infer_shape, read_csv
from generator import generate_system
from regressors import SKRegressor


def generate_training_data(name):
    csv_path = name + "-training.csv"
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    # this system is promising, HOPA reaches slack=-0.0028144504861013437
    random = Random(123)
    shape = (4, 5, 3)
    n_flows, t_tasks, n_procs = shape
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=0.84,
                             period_min=100,
                             period_max=100*3,
                             deadline_factor_min=0.5,
                             deadline_factor_max=1)

    # callback writes the result of each HOPA iteration to a csv file
    def csv_writer_cb(s):
        nonlocal shape, csv_path
        write_to_csv(s, shape, csv_path, normalize=True)

    analysis = HolisticAnalyis(limit_factor=10, reset=False)
    hopa = HOPAssignment(analysis=analysis, verbose=True, callback=csv_writer_cb)
    system.apply(hopa)

    # enhance the training data with random walk of priorities and processors
    def csv_writer_analysis_cb(s):
        nonlocal shape, csv_path, analysis
        s.apply(analysis)
        write_to_csv(s, shape, csv_path, normalize=True)

    walk_random_priorities_processors(system, 100, 100, csv_writer_analysis_cb, verbose=True)

    return system, csv_path


def train_model(name, csv_path):
    df = read_csv(csv_path)
    X, y = get_xy(df, "label_slack_system")

    model = MLPRegressor(random_state=1, max_iter=200, verbose=True, hidden_layer_sizes=(1000, 1000,))
    model.fit(X, y)
    return model


def generate_testing_data(name, system):
    csv_path = name + "-testing.csv"
    shape = infer_shape([system])
    analysis = HolisticAnalyis(limit_factor=10, reset=False)

    def test_cb(s):
        nonlocal shape, analysis, csv_path
        s.apply(analysis)
        write_to_csv(s, shape, csv_path, normalize=True)

    walk_random_priorities(system, 10, 10, test_cb, verbose=True)
    return csv_path


def test_model(name, model, csv_testing):
    df = read_csv(csv_testing)
    X, y = get_xy(df, "label_slack_system")
    y_pred = model.predict(X)
    plt.scatter(y, y_pred)
    plt.show()


if __name__ == '__main__':
    name = Path(__file__).stem

    system, csv_training = generate_training_data(name)
    csv_testing = generate_testing_data(name, system)

    model = train_model(name, csv_training)
    test_model(name, model, csv_testing)
