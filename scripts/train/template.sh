export HF_MODEL_DIR="/zouxiangyu/codes/Learning/In-Place-TTT/outputs/longsft-swa-full-halo-1to3-kd-swa4096-chunk1024-4k/hf_global_step_8000" && \
export OUTPUT_DIR="/zouxiangyu/codes/Learning/In-Place-TTT/outputs/longsft-swa-anchor-ttt-aux-swa4096-chunk1024-32k-kdinit-gs8000" && \
export WANDB_NAME="longsft-swa-anchor-ttt-aux-swa4096-chunk1024-32k-kdinit-gs8000" && \
bash "scripts/train/longsft/run_swa_anchor_ttt_aux.sh" \
--model.model_path "$HF_MODEL_DIR" \
--model.config_path "$HF_MODEL_DIR" \
--model.tokenizer_path "$HF_MODEL_DIR" \
--train.load_checkpoint_path ""