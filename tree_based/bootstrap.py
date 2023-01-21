import time

import numpy as np
import pandas as pd
from sklearn.utils import resample
from lifelines.utils import concordance_index as ci_lifelines

from logger import get_logger
from utils import msg_box, consuming_time


def bootstrapping(event_times, y_test, y_pred, y_prob, metrics_pred, metrics_prob, repeat=1000):
    logger = get_logger("Bootstrapping")
    msg = msg_box("Bootstrap")
    msg += f"\nBootstrap for {repeat} times..."
    logger.info(msg)

    results = pd.DataFrame()
    start = time.time()
    for number in range(repeat):
        ids = np.arange(len(y_test))
        sample_id = resample(ids)
        event_times_ = event_times[sample_id]
        y_test_ = y_test[sample_id]
        y_pred_ = y_pred[sample_id]
        y_prob_ = y_prob[sample_id]

        metrics = [*metrics_pred.keys()] + [*metrics_prob.keys()]
        metrics_result = pd.DataFrame(index=metrics)
        for key, value in metrics_pred.items():
            metrics_result.at[key, ""] = value(y_test_, y_pred_)
        for key, value in metrics_prob.items():
            metrics_result.at[key, ""] = value(y_test_, y_prob_)
        # compute c_index
        metrics_result.at["c_index", ""] = ci_lifelines(event_times, -y_prob_, y_test_)

        metrics_result = metrics_result[""].rename(number)
        results = pd.concat((results, metrics_result), axis=1)

    msg = msg_box("result")

    end = time.time()
    total_time = consuming_time(start, end)
    msg += f"\nConsuming time: {total_time}."

    boot_result = pd.DataFrame()
    boot_result["CI_median"] = results.median(axis=1)
    boot_result["CI_low"] = results.quantile(q=0.025, axis=1)
    boot_result["CI_high"] = results.quantile(q=0.975, axis=1)
    boot_result["CI_half"] = (boot_result["CI_high"] - boot_result["CI_low"]) / 2
    msg += f"\n{boot_result}"

    logger.info(msg)
