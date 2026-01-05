#!/bin/bash
# Automatically fix Python shebangs to use conda environment after colcon build

CONDA_ENV="foundationpose_ros"
CONDA_PYTHON="/home/ehsanullahm1/miniconda3/envs/${CONDA_ENV}/bin/python3"
INSTALL_DIR="$(pwd)/install/foundationpose_bridge/lib/foundationpose_bridge"

echo "Fixing shebangs to use: $CONDA_PYTHON"

# Find all entry point scripts and fix their shebangs
for script in "$INSTALL_DIR"/*; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        # Check if it has a python shebang
        if head -1 "$script" | grep -q "^#!/usr/bin/python"; then
            echo "Fixing: $script"
            # Use sed to replace the shebang
            sed -i "1s|^#!/usr/bin/python.*|#!${CONDA_PYTHON}|" "$script"
        fi
    fi
done

echo "Done! All shebangs fixed."
