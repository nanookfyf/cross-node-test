git clone -b flash_epscale-cleanup-pause-fix https://github.com/nanookfyf/vllm
cd vllm
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -v
pip install meson ninja pybind11 tomlkit
pip install nixl-cu13==1.0.1
pip install ray[default]