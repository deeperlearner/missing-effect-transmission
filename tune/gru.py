from logger import get_logger
from mains import train, train_mp
from parse_config import ConfigParser
from utils import msg_box, consuming_time, get_by_path

objective_results = []


def objective(trial, CNN=False, RNN=True, NN=False):
    # TODO: hyperparameters search spaces
    modification = {}

    # # batch_size
    # batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    # data_loaders_mod = {
    #     "data_loaders;train;data;kwargs;DataLoader_kwargs;batch_size": batch_size,
    # }
    # modification.update(data_loaders_mod)

    # models
    min_units = 16
    max_units = 64
    step = int((max_units - min_units) / 4)
    ## CNN_params
    if CNN:
        cnn_layers = trial.suggest_int("cnn_layers", 1, 3)
        cnn_hiddens = [trial.suggest_int(f"cnn_units_{i}", min_units, max_units, step=step)
                       for i in range(0, cnn_layers)]
        conv_kernel = trial.suggest_int("conv_kernel", 1, 5)
        pool_kernel = trial.suggest_int("pool_kernel", 1, 3)
        pool_stride = trial.suggest_int("pool_stride", 1, 3)
        CNN_mod = {
            "models;model;kwargs;CNN_params;hiddens": cnn_hiddens,
            "models;model;kwargs;CNN_params;conv_kernel": conv_kernel,
            "models;model;kwargs;CNN_params;pool_kernel": pool_kernel,
            "models;model;kwargs;CNN_params;pool_stride": pool_stride,
        }
        modification.update(CNN_mod)

    ## RNN_params
    if RNN:
        hidden_size = trial.suggest_int("hidden_size", min_units, max_units, step=step)
        RNN_dropout = trial.suggest_uniform("RNN_dropout", 0.1, 0.4)
        RNN_mod = {
            "models;model;kwargs;RNN_params;hidden_size": hidden_size,
            "models;model;kwargs;RNN_params;dropout_rate": RNN_dropout,
        }
        modification.update(RNN_mod)
    ## NN_multi_params
    else:
        hidden_size = 0
        min_units = 64
        max_units = 1024
        step = int((max_units - min_units) / 4)
        n_fc_layers = trial.suggest_int("n_multi", 2, 5)
        fc_hiddens = [trial.suggest_int(f"fc_multi_{i}", min_units, max_units, step=step)
                      for i in range(0, n_fc_layers)]
        NN_multi_dropout = trial.suggest_uniform("NN_multi_dropout", 0.1, 0.4)
        NN_multi_mod = {
            "models;model;kwargs;NN_multi_params;hiddens": fc_hiddens,
            "models;model;kwargs;NN_multi_params;dropout_rate": NN_multi_dropout,
        }
        modification.update(NN_multi_mod)

    ## NN_params
    if NN:
        # min_units = 16
        # max_units = 64
        # step = int((max_units - min_units) / 4)
        # n_fc_layers = trial.suggest_int("n_fc_layers", 2, 5)
        # fc_hiddens = [trial.suggest_int(f"fc_units_{i}", min_units, max_units, step=step)
        #               for i in range(0, n_fc_layers)]
        nn_down_factor = trial.suggest_int("nn_down_factor", 1, 3)
        nn_dropout = trial.suggest_uniform("nn_dropout", 0.1, 0.4)
        NN_mod = {
            "models;model;kwargs;NN_params;down_factor": nn_down_factor,
            "models;model;kwargs;NN_params;dropout_rate": nn_dropout,
        }
        modification.update(NN_mod)

    # optimizers
    opt_type = trial.suggest_categorical("opt_type", ["Adam", "SGD"])
    lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
    l2_value = trial.suggest_loguniform("l2_value", 1e-4, 1e-1)
    optimizers_mod = {
        "optimizers;model;type": opt_type,
        "optimizers;model;kwargs;lr": lr,
        "optimizers;model;kwargs;weight_decay": l2_value,
    }
    modification.update(optimizers_mod)

    config = ConfigParser(modification)
    logger = get_logger("optuna")
    keys = ["trainers", "trainer", "kwargs", "monitor"]
    max_min, mnt_metric = get_by_path(config, keys).split()
    k_fold = config["cross_validation"]["k_fold"]
    msg = msg_box("Optuna progress")
    i, N = len(objective_results), config["optuna"]["n_trials"]
    msg += f"\ntrial: ({i}/{N-1})"
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
