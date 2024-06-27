source activate melody_training
NVIDIA_PACKAGE_DIR="/opt/conda/envs/melody_training/lib/python3.10/site-packages/nvidia"

for dir in $NVIDIA_PACKAGE_DIR/*; do
    if [ -d "$dir/lib" ]; then
        export LD_LIBRARY_PATH="$dir/lib:$LD_LIBRARY_PATH"
    fi
done

python train.py --data-dir './data' --config-path './config.json'