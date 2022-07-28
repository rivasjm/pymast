import os
import pathlib
from random import Random
from pathlib import Path

import numpy

from analysis import HolisticAnalyis
from assignment import HOPAssignment, walk_random_priorities
from data import write_to_csv, get_xy, infer_shape
from generator import generate_system
from regressors import SKRegressor


def generate_csv():
    name = Path(__file__).stem
    csv_path = name + ".csv"
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
    return system


def create_model():
    import pandas as pd
    name = Path(__file__).stem
    csv_path = name + ".csv"

    df: pd.DataFrame = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    X, y = get_xy(df, "label_slack_system")

    from sklearn.neural_network import MLPRegressor
    reg = MLPRegressor(random_state=1, max_iter=200, verbose=True, hidden_layer_sizes=(10000, 10000,))
    reg.fit(X, y)

    from joblib import dump
    dump(reg, name + ".joblib")
    return reg


def apply_model(model, system):
    shape = infer_shape([system])
    regressor = SKRegressor(model=model, shape=shape)

    def predic_cb(s):
        nonlocal regressor
        prediction = regressor.predict(s)
        print(prediction)

    walk_random_priorities(system, 10, 2, predic_cb)


if __name__ == '__main__':
    system = generate_csv()
    reg = create_model()
    apply_model(reg, system)
