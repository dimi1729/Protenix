import torch
# ckpt = torch.load("/af3-dev/release_model/model_v1.pt", map_location="cpu")
# ckpt = torch.load("/af3-dev/release_model/model_v1.pt")
# ckpt = torch.load(
#     "/mnt/bn/ai4sml/prod/evaluate_ckpts/retrain-confidence-from-v7ckpt-88229-96-acc1-lr0-0005_20241123_013512/checkpoints/9199_ema_0.999.pt",
#     map_location=torch.device("cpu"),
# )
# for k in ["optimizer", "scheduler", "step"]:
#     del ckpt[k]
# dst_path = "/af3-dev/release_model/model_v2.pt"
# torch.save(ckpt, dst_path)
ckpt = torch.load(
    "/mnt/bn/ai4s-yg/zhangyuxuan/protenix/release_data/checkpoint/protenix_mini_default_v0.5.0.pt",
    map_location=torch.device("cpu"),
)
for k in ["optimizer", "scheduler", "step"]:
    del ckpt[k]
def replace_weights_in_state_dict(state_dict, key, source_indices, target_indices):
    if key in state_dict:
        with torch.no_grad():
            weights = state_dict[key].clone()
            weights[..., target_indices] = weights[..., source_indices]
            state_dict[key] = weights
    return state_dict
def replace_msa_params_in_state_dict(state_dict, prefix=""):
    state_dict = replace_weights_in_state_dict(
        state_dict=state_dict,
        key=prefix + "msa_module.linear_no_bias_m.weight",
        source_indices=[29, 25, 26, 27, 28],
        target_indices=[25, 26, 27, 28, 29],
    )
    state_dict = replace_weights_in_state_dict(
        state_dict=state_dict,
        key=prefix + "linear_no_bias_sinit.weight",
        source_indices=[
            384 + 32 + 29,
            384 + 32 + 25,
            384 + 32 + 26,
            384 + 32 + 27,
            384 + 32 + 28,
        ],
        target_indices=[
            384 + 32 + 25,
            384 + 32 + 26,
            384 + 32 + 27,
            384 + 32 + 28,
            384 + 32 + 29,
        ],
    )
    state_dict = replace_weights_in_state_dict(
        state_dict=state_dict,
        key=prefix + "msa_module.linear_no_bias_s.weight",
        source_indices=[
            384 + 32 + 29,
            384 + 32 + 25,
            384 + 32 + 26,
            384 + 32 + 27,
            384 + 32 + 28,
        ],
        target_indices=[
            384 + 32 + 25,
            384 + 32 + 26,
            384 + 32 + 27,
            384 + 32 + 28,
            384 + 32 + 29,
        ],
    )
    state_dict = replace_weights_in_state_dict(
        state_dict=state_dict,
        key=prefix + "diffusion_module.diffusion_conditioning.layernorm_s.weight",
        source_indices=[
            384 + 384 + 32 + 29,
            384 + 384 + 32 + 25,
            384 + 384 + 32 + 26,
            384 + 384 + 32 + 27,
            384 + 384 + 32 + 28,
        ],
        target_indices=[
            384 + 384 + 32 + 25,
            384 + 384 + 32 + 26,
            384 + 384 + 32 + 27,
            384 + 384 + 32 + 28,
            384 + 384 + 32 + 29,
        ],
    )
    state_dict = replace_weights_in_state_dict(
        state_dict=state_dict,
        key=prefix + "diffusion_module.diffusion_conditioning.linear_no_bias_s.weight",
        source_indices=[
            384 + 384 + 32 + 29,
            384 + 384 + 32 + 25,
            384 + 384 + 32 + 26,
            384 + 384 + 32 + 27,
            384 + 384 + 32 + 28,
        ],
        target_indices=[
            384 + 384 + 32 + 25,
            384 + 384 + 32 + 26,
            384 + 384 + 32 + 27,
            384 + 384 + 32 + 28,
            384 + 384 + 32 + 29,
        ],
    )
    state_dict = replace_weights_in_state_dict(
        state_dict=state_dict,
        key=prefix + "confidence_head.linear_no_bias_s1.weight",
        source_indices=[
            384 + 32 + 29,
            384 + 32 + 25,
            384 + 32 + 26,
            384 + 32 + 27,
            384 + 32 + 28,
        ],
        target_indices=[
            384 + 32 + 25,
            384 + 32 + 26,
            384 + 32 + 27,
            384 + 32 + 28,
            384 + 32 + 29,
        ],
    )
    state_dict = replace_weights_in_state_dict(
        state_dict=state_dict,
        key=prefix + "confidence_head.linear_no_bias_s2.weight",
        source_indices=[
            384 + 32 + 29,
            384 + 32 + 25,
            384 + 32 + 26,
            384 + 32 + 27,
            384 + 32 + 28,
        ],
        target_indices=[
            384 + 32 + 25,
            384 + 32 + 26,
            384 + 32 + 27,
            384 + 32 + 28,
            384 + 32 + 29,
        ],
    )
    return state_dict
sample_key = [k for k in ckpt["model"].keys()][0]
print(sample_key)
ckpt["model"] = replace_msa_params_in_state_dict(
    ckpt["model"],
    prefix="module." if sample_key.startswith("module.") else "",
)
dst_path = (
    "/mnt/bn/ai4s-yg/chengyuegong/checkpoints/protenix_mini_default_v0.5.0.pt"
)
torch.save(ckpt, dst_path)