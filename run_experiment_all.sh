#!/bin/bash

set -e

BASE_SCRIPT="./run_experiment.sh"
#BASE_SCRIPT="echo"
BASE_APP="./experiments/train_individual.py"

COMMON_OPTS=(--net KerasNet --ds CIFAR-10 --opt adam --seed 42 --bs 128 --dev gpu)
COMMON_OPTS+=("--wandb")

echo "---------------- Stage 1 ----------------"
"$BASE_SCRIPT" "$BASE_APP" base --acts ReLU --start_ep 0 --end_ep 100 "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" base --acts SiLU --start_ep 0 --end_ep 100 "${COMMON_OPTS[@]}"

echo "---------------- Stage 2 ----------------"
"$BASE_SCRIPT" "$BASE_APP" ahaf_shared --acts ReLU \
                --start_ep 100 --end_ep 200 --tune_aaf \
                --patch_base \
                "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts ReLU \
                --start_ep 100 --end_ep 200 --tune_aaf --p24sl \
                --patch_base \
                "${COMMON_OPTS[@]}"

echo "---------------- Stage 3 ----------------"
"$BASE_SCRIPT" "$BASE_APP" ahaf_shared --acts SiLU \
                --start_ep 100 --end_ep 200 \
                --patch_base "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts SiLU \
                --start_ep 100 --end_ep 200 --p24sl --tune_aaf \
                --patch_base "${COMMON_OPTS[@]}"

echo "---------------- Stage 4 ----------------"
"$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn_shared --acts Tanh --act_cnn ReLU \
               --start_ep 100 --end_ep 500 --tune_aaf \
               --patch_base --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts Tanh --act_cnn ReLU \
               --start_ep 100 --end_ep 500 --tune_aaf --p24sl \
               --patch_base --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn_shared --acts Random --act_cnn ReLU \
               --start_ep 100 --end_ep 500 --tune_aaf \
               --patch_base --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"

echo "---------------- Stage 5 ----------------"
"$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn_shared --acts Tanh --act_cnn SiLU \
               --start_ep 100 --end_ep 500 --tune_aaf \
               --patch_base --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts Tanh --act_cnn SiLU \
               --start_ep 100 --end_ep 500 --tune_aaf --p24sl \
               --patch_base --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" fuzzy_ffn_shared --acts Random --act_cnn SiLU \
               --start_ep 100 --end_ep 500 --tune_aaf \
               --patch_base --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
if false; then
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts Tanh --act_cnn SiLU \
               --start_ep 200 --end_ep 500 --tune_aaf --tuned --p24sl \
               --patched --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
fi

echo "---------------- Stage 6 ----------------"
"$BASE_SCRIPT" "$BASE_APP" ahaf_shared --acts ReLU \
               --start_ep 100 --end_ep 500 --tune_aaf \
               --patch_base --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts ReLU \
               --start_ep 100 --end_ep 500 --tune_aaf --p24sl \
               --patch_base --patched_from_af SiLU \
               "${COMMON_OPTS[@]}"

echo "---------------- Stage 7 ----------------"
"$BASE_SCRIPT" "$BASE_APP" ahaf_shared --acts SiLU \
               --start_ep 100 --end_ep 500 --tune_aaf \
               --patch_base --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"
"$BASE_SCRIPT" "$BASE_APP" leaf_shared --acts SiLU \
               --start_ep 100 --end_ep 500 --tune_aaf --p24sl \
               --patch_base --patched_from_af ReLU \
               "${COMMON_OPTS[@]}"
