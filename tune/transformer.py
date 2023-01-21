from logger import get_logger
from mains import train, train_mp
from parse_config import ConfigParser
from utils import msg_box, consuming_time

objective_results = []


def objective(trial):
    # TODO: hyperparameters search spaces

    # # batch_size
    # batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # models
    ## encoder
    nhead = 1
    nhid = trial.suggest_int(f"nhid", 64, 256, step=16)
    nlayers = trial.suggest_int("nlayers", 1, 3)
    enc_dropout = trial.suggest_uniform("enc_dropout", 0.1, 0.4)
    ## NN_params
    min_units = 16
    max_units = 64
    step = int((max_units - min_units) / 4)
    n_fc_layers = trial.suggest_int("n_fc_layers", 2, 5)
    fc_hiddens = [trial.suggest_int(f"fc_units_{i}", min_units, max_units, step=step)
                  for i in range(0, n_fc_layers)]
    nn_dropout = trial.suggest_uniform("nn_dropout", 0.1, 0.4)

    # optimizers
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    l2_value = trial.suggest_loguniform("l2_value", 1e-5, 1e-1)

    modification = {
        # "data_loaders;train;data;kwargs;DataLoader_kwargs;batch_size": batch_size,
        "models;model;kwargs;nhead": nhead,
        "models;model;kwargs;nhid": nhid,
        "models;model;kwargs;nlayers": nlayers,
        "models;model;kwargs;dropout": enc_dropout,
        "models;model;kwargs;NN_params;hiddens": fc_hiddens,
        "models;model;kwargs;NN_params;dropout_rate": nn_dropout,
        "optimizers;model;kwargs;lr": lr,
        "optimizers;model;kwargs;weight_decay": l2_value,
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
