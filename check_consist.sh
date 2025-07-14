export LAYERNORM_TYPE=openfold_layernorm

N_sample=5
N_step=2
N_cycle=4
seed=101
use_deepspeed_evo_attention=false
input_json_path="/mnt/bn/ai4s-yg/zhangyuxuan/protenix/examples/example2.json"
dump_dir="/mnt/bn/ai4s-yg/chengyuegong/fix_bf16_output_esm_protenix"
python3 runner/inference.py \
--use_deepspeed_evo_attention false \
--load_checkpoint_dir "/mnt/bn/ai4s-yg/chengyuegong/checkpoints" \
--model_name "protenix_mini_esm_v0.5.0" \
--dtype bf16 \
--seeds ${seed} \
--dump_dir ${dump_dir} \
--input_json_path ${input_json_path} \
--model.N_cycle ${N_cycle} \
--sample_diffusion.N_sample ${N_sample} \
--sample_diffusion.N_step ${N_step} \
--sample_diffusion.gamma0 0. \
--sample_diffusion.step_scale_eta 1. \
--load_strict true \
--infer_setting.chunk_size 256 \
--num_workers 0 \
--data.msa.min_size.test 2000 \
--data.msa.sample_cutoff.test 2000 \
--deterministic true \
--sorted_by_ranking_score false \
