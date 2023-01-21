from logger import get_logger
from mains import train, train_mp
from parse_config import ConfigParser
from utils import msg_box, consuming_time, get_by_path

objective_results = []


def objective(trial, **kwargs):
    # TODO: hyperparameters search spaces
    # RUS_rate = trial.suggest_int("RUS_rate", 20, 40)
    nn_dropout = trial.suggest_uniform("nn_dropout", 0.1, 0.6)
    len_epoch = trial.suggest_int("len_epoch", 100, 233)

    modification = {
        # "data_loaders;train;data;kwargs;RUS_rate": RUS_rate,
        "models;model;kwargs;NN_params;dropout_rate": nn_dropout,
        "trainers;trainer;kwargs;len_epoch": len_epoch,
    }
    config = ConfigParser(modification=modification)
    logger = get_logger("optuna")
    keys = ["trainers", "trainer", "kwargs", "monitor"]
    max_min, mnt_metric = get_by_path(config, keys).split()
    k_fold = config["cross_validation"]["k_fold"]
    msg = msg_box("Optuna progress")
    i, N = len(objective_results), config["optuna"]["n_trials"]
    msg += f"\nTrial: ({i}/{N-1})"
    logger.info(msg)

    if config.run_args.mp:
        # train with multiprocessing on k_fold
        results = train_mp(config)
        result = sum(results) / len(results)
    else:
        result = train(config)
    objective_results.append(result)

    config.set_log(log_name="optuna.log")
    if (
        max_min == "max"
        and result >= max(objective_results)
        or max_min == "min"
        and result <= min(objective_results)
    ):
        msg = "Backing up best hyperparameters config and model..."
        config.backup(best_hp=True)
        config.cp_models()
        logger.info(msg)

    return result
