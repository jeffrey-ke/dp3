# bash scripts/gen_demonstration_adroit.sh door
# bash scripts/gen_demonstration_adroit.sh hammer
# bash scripts/gen_demonstration_adroit.sh pen

# build 3d diffusion policy to compile any changes made to the code
cd 3D-Diffusion-Policy
pip install -e .
cd ../

cd third_party/VRL3/src

task=${1}

CUDA_VISIBLE_DEVICES=0 python gen_demonstration_expert.py --env_name $task \
                        --num_episodes 10 \
                        --root_dir "../../../3D-Diffusion-Policy/data/" \
                        --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop
