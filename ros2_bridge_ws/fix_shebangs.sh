#!/bin/bash
# Automatically fix Python shebangs to use conda environment after colcon build

CONDA_ENV="foundationpose_ros"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    echo "ERROR: Activate the '$CONDA_ENV' conda environment first." >&2
    exit 1
fi

CONDA_PYTHON="$(command -v python3)"
if [[ ! -x "$CONDA_PYTHON" ]]; then
    echo "ERROR: Could not resolve the active Python interpreter." >&2
    exit 1
fi
INSTALL_DIR="$(pwd)/install/foundationpose_bridge/lib/foundationpose_bridge"

echo "Fixing shebangs to use: $CONDA_PYTHON"

# Find all entry point scripts and fix their shebangs
for script in "$INSTALL_DIR"/*; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        # Check if it has a python shebang
        if head -1 "$script" | grep -qE '^#!.*python'; then
            echo "Fixing: $script"
            # Use sed to replace the shebang
            sed -i "1c\\#!${CONDA_PYTHON}" "$script"
        fi
    fi
done

echo "Done! All shebangs fixed."
