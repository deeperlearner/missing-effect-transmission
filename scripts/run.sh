#!/usr/bin/env bash
# ------------------
#  PyTorch Template
# ------------------
# Repository    : https://github.com/deeperlearner/pytorch-template
VERSION="v1.0.0"


# This script run train and test
usage() { echo "Usage: $0 [-abcde]" 1>&2; exit 1; }

# record execution time to log
time_log() {
    RUNNING_TIME=$(($SECONDS/86400))" days, "$(date +%T -d "1/1 + $SECONDS sec")
    echo -e "---------------------------------" | tee -a $LOG_FILE
    echo -e "$TYPE running time: $RUNNING_TIME" | tee -a $LOG_FILE
    let "TOTAL_SECONDS += $SECONDS"
}


mkdir -p log
LOG_FILE="log/run.log"
echo "===============================" >> $LOG_FILE
echo "date: $(date)" >> $LOG_FILE
echo "version: $VERSION" >> $LOG_FILE
TOTAL_SECONDS=0
DATA_PATH="./data"
# "CONFIG##*/" is the basename of CONFIG
while getopts "abcde" flag; do
  case "$flag" in
    a)
      SECONDS=0
      TYPE="debug"

      CONFIG="dataset_model"
      EXP="dataset_model"
      RUN_ID=${VERSION}

      # python3 data_loaders/ehr_dataset.py
      # # use optuna to find the best h.p.
      # python3 mains/main.py --optuna --mp -c "configs/$CONFIG.json" --mode train \
      #     --run_id $RUN_ID --name $EXP
      # python3 mains/main.py -c "saved/$EXP/$RUN_ID/best_hp/${CONFIG##*/}.json" --mode test \
      #     --resume "saved/$EXP/$RUN_ID/best_hp/fold_0_model_best.pth" --run_id $RUN_ID --k_fold 1 --bootstrapping

      # # given h.p. with k_fold = 1
      # python3 mains/main.py -c "configs/$CONFIG.json" --mode train \
      #     --run_id $RUN_ID --name $EXP --k_fold 1
      # python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test \
      #     --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID --bootstrapping

      time_log
      ;;
    b)
      SECONDS=0
      TYPE="preprocess"

      jupyter-notebook prepare-data.ipynb

      time_log
      ;;
    c)
      SECONDS=0
      TYPE="optuna"

      CONFIG="bert_half_head"
      EXP="bert_half_head"
      RUN_ID="optuna"
      # use optuna to find the best h.p.
      python3 mains/main.py --optuna -c "configs/$CONFIG.json" --mode train \
          --data_dir $DATA_PATH \
          --run_id $RUN_ID --name $EXP
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/best_hp/${CONFIG##*/}.json" --mode test \
          --data_dir $DATA_PATH \
          --resume "saved/$EXP/$RUN_ID/best_hp/fold_0_model_best.pth" --run_id $RUN_ID --k_fold 1 --bootstrapping

      time_log
      ;;
    d)
      SECONDS=0
      TYPE="baselines"

      CONFIG="dataset_model"
      EXP="dataset_model"
      RUN_ID="run"
      python3 mains/main.py -c "configs/$CONFIG.json" --mode train \
          --data_dir $DATA_PATH \
          --run_id $RUN_ID --name $EXP --k_fold 1
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test \
          --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID --bootstrapping

      time_log
      ;;
    e)
      SECONDS=0
      TYPE="bert"

      # CONFIG="tree_based"
      # EXP="RF"
      # RUN_ID="run"
      # python3 tree_based/main.py -c "configs/$CONFIG.json" --mode train \
      #     --data_dir $DATA_PATH \
      #     --run_id $RUN_ID --name $EXP --k_fold 1 --last_N 1
      # python3 tree_based/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test \
      #     --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID --name $EXP --ds_mode test --bootstrapping

      # CONFIG="gru"
      # EXP="gru"
      # RUN_ID="run"
      # python3 mains/main.py -c "configs/$CONFIG.json" --mode train \
      #     --data_dir $DATA_PATH \
      #     --run_id $RUN_ID --name $EXP --k_fold 1
      # python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test --ds_mode test \
      #     --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID --bootstrapping

      CONFIG="bert"
      EXP="bert"
      RUN_ID="TransEHR_sin_pos"
      python3 mains/main.py -c "configs/$CONFIG.json" --mode train \
          --data_dir $DATA_PATH \
          --run_id $RUN_ID --name $EXP --k_fold 1
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test --ds_mode test \
          --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID #--bootstrapping

      CONFIG="bert_half_head"
      EXP="bert_half_head"
      RUN_ID="half_sin_pos"
      python3 mains/main.py -c "configs/$CONFIG.json" --mode train \
          --data_dir $DATA_PATH \
          --run_id $RUN_ID --name $EXP --k_fold 1
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test --ds_mode test \
          --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID #--bootstrapping

      time_log
      ;;
    *)
      usage
      ;;
  esac
done

TOTAL_TIME=$(($TOTAL_SECONDS/86400))" days, "$(date +%T -d "1/1 + $TOTAL_SECONDS sec")
echo -e "---------------------------------" | tee -a $LOG_FILE
echo -e "total running time: $TOTAL_TIME" | tee -a $LOG_FILE
