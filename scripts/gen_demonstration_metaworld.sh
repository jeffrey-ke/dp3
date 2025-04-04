# bash scripts/gen_demonstration_metaworld.sh basketball


# all v2_envs here: https://github.com/Farama-Foundation/Metaworld/blob/304d79db1fd56401aff00a92181db0c977a04663/metaworld/envs/mujoco/env_dict.py#L214
cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10 \
            --root_dir "../../3D-Diffusion-Policy/data/" \
