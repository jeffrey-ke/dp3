# bash scripts/gen_demonstration_adroit.sh door
# bash scripts/gen_demonstration_adroit.sh hammer
# bash scripts/gen_demonstration_adroit.sh pen

cd third_party/VRL3/src

task=${1}
feature_layer=${2:-None}
policy_name=${3}
gpu_id=${4:-0}
CUDA_VISIBLE_DEVICES=$gpu_id python gen_demonstration_expert_custom_res.py --env_name $task \
                        --policy_name $policy_name \
                        --num_episodes 10 \
                        --root_dir "../../../3D-Diffusion-Policy/data/" \
                        --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop \
                        --feature_layer $feature_layer
