ml --force purge
ml Stages/2024 GCCcore/.12.3.0 Python/3.11.3 

ml NCCL/default-CUDA-12 PyTorch/2.1.2 torchvision/0.16.2

python3 -m venv ray_juwels_env

source ray_juwels_env/bin/activate

pip3 install ray ray[tune]
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
pip3 install glob2

deactivate