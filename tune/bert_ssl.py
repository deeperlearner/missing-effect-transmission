from logger import get_logger
from mains import train, train_mp
from parse_config import ConfigParser
from utils import msg_box, consuming_time

objective_results = []


def objective(trial):
    # TODO: hyperparameters search spaces

    # optimizers
    lr_M = trial.suggest_float("lr_M", 1e-5, 1, log=True)
    l2_M = trial.suggest_loguniform('l2_M', 1e-5, 1e-1)
    # lr_X = trial.suggest_float("lr_X", 1e-5, 1, log=True)
    # l2_X = trial.suggest_loguniform('l2_X', 1e-5, 1e-1)
    lr_y = trial.suggest_float("lr_y", 1e-5, 1, log=True)
    l2_y = trial.suggest_loguniform('l2_y', 1e-5, 1e-1)

    # lr_schedulers
    gamma_M = trial.suggest_float("gamma_M", 0.5, 1)
    gamma_X = trial.suggest_float("gamma_X", 0.5, 1)
    gamma_y = trial.suggest_float("gamma_y", 0.5, 1)

    modification = {
        "optimizers;mask_decoder;kwargs;lr": lr_M,
        "optimizers;mask_decoder;kwargs;weight_decay": l2_M,
        # "optimizers;recon_decoder;kwargs;lr": lr_X,
        # "optimizers;recon_decoder;kwargs;weight_decay": l2_X,
        "optimizers;target_decoder;kwargs;lr": lr_y,
        "optimizers;target_decoder;kwargs;weight_decay": l2_y,
        "lr_schedulers;mask_decoder;kwargs;gamma": gamma_M,
        # "lr_schedulers;recon_decoder;kwargs;gamma": gamma_X,
        "lr_schedulers;target_decoder;kwargs;gamma": gamma_y,
    }

    config = ConfigParser(modification)
    logger = get_logger("optuna")
    max_min, mnt_metric = config["trainer"]["kwargs"]["monitor"].split()
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
