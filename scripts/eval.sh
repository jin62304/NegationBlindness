######## free-form generation, trivia, chatgpt ########
MOD_TYPE=chatgpt
VAL_TYPE=free
DATA_TYPE=trivia
PROMPT=vanilla

# pos
CUDA_VISIBLE_DEVICES=5 python3 main.py \
  --task "eval" \
  --model_type "${MOD_TYPE}" \
  --data_type "${DATA_TYPE}" \
  --eval_model "gpt4" \
  --prompt "${PROMPT}" \
  --val_type "${VAL_TYPE}"
# neg
CUDA_VISIBLE_DEVICES=5 python3 main.py \
  --task "eval" \
  --model_type "${MOD_TYPE}" \
  --data_type "${DATA_TYPE}" \
  --eval_model "gpt4" \
  --prompt "${PROMPT}" \
  --val_type "${VAL_TYPE}" \
  --cor_type "negation"
#######################################################