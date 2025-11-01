######## free-form generation, trivia, chatgpt ########
# pos
CUDA_VISIBLE_DEVICES=5 python3 main.py \
  --task "gen" \
  --model_type "chatgpt" \
  --data_type "trivia" \
  --prompt "vanilla" \
  --val_type "free"
# neg
CUDA_VISIBLE_DEVICES=5 python3 main.py \
  --task "gen" \
  --model_type "chatgpt" \
  --data_type "trivia" \
  --prompt "vanilla" \
  --val_type "free" \
  --cor_type "negation"
#######################################################

######## free-form generation, trivia, hf model (non-API) ########
MODEL_TYPE=mistralai/Mistral-7B-Instruct-v0.2
#MODEL_TYPE=Qwen2.5-1.5B-Instruct
# pos
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
  --task "gen" \
  --model_type ${MODEL_TYPE} \
  --data_type "trivia" \
  --prompt "vanilla" \
  --val_type "free"
# neg
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
  --task "gen" \
  --model_type ${MODEL_TYPE} \
  --data_type "trivia" \
  --prompt "vanilla" \
  --val_type "free" \
  --cor_type "negation"
###################################################################