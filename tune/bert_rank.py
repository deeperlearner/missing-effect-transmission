from logger import get_logger
from mains import train, train_rank_mp
from parse_config import ConfigParser
from utils import msg_box, consuming_time, get_by_path

objective_results = []


def objective(trial):
    # TODO: hyperparameters search spaces
    modification = {}

    # # batch_size
    # batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    # data_loaders_mod = {
    #     "data_loaders;train;data;kwargs;DataLoader_kwargs;batch_size": batch_size,
    # }
    # modification.update(data_loaders_mod)

    # models
    ## BERT encoder
    # d_model = trial.suggest_int("d_model", 64, 256, step=16)
    # nhead = trial.suggest_int("nhead", 1, 2)
    # nhid = trial.suggest_int("nhid", 64, 256, step=16)
    # nlayers = trial.suggest_int("nlayers", 2, 6)
    # enc_dropout = trial.suggest_uniform("enc_dropout", 0.1, 0.5)
    ## NN_params
    # min_units = 16
    # max_units = 64
    # step = int((max_units - min_units) / 4)
    # n_fc_layers = trial.suggest_int("n_fc_layers", 2, 5)
    # fc_hiddens = [trial.suggest_int(f"fc_units_{i}", min_units, max_units, step=step)
    #               for i in range(0, n_fc_layers)]
    # nn_down_factor = trial.suggest_int("nn_down_factor", 1, 5)
    # nn_dropout = trial.suggest_uniform("nn_dropout", 0.1, 0.5)
    # models_mod = {
    #     "models;model;kwargs;d_model": d_model,
    #     "models;model;kwargs;nhead": nhead,
    #     "models;model;kwargs;nhid": nhid,
    #     "models;model;kwargs;nlayers": nlayers,
    #     "models;model;kwargs;dropout": enc_dropout,
    #     "models;model;kwargs;NN_params;down_factor": nn_down_factor,
    #     "models;model;kwargs;NN_params;dropout_rate": nn_dropout,
    # }
    # modification.update(models_mod)

    # losses
    alpha = trial.suggest_uniform("alpha", 1.0, 10.0)
    losses_mod = {
        "losses;rank_loss;kwargs;alpha": alpha,
    }
    modification.update(losses_mod)

    # optimizers
    # opt_type = trial.suggest_categorical("opt_type", ["Adam", "SGD"])
    # lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
    # l2_value = trial.suggest_loguniform("l2_value", 1e-4, 1e-1)
    # optimizers_mod = {
    #     "optimizers;model;type": opt_type,
    #     "optimizers;model;kwargs;lr": lr,
    #     "optimizers;model;kwargs;weight_decay": l2_value,
    # }
    # modification.update(optimizers_mod)

    config = ConfigParser(modification)
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
        results = train_rank_mp(config)
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
