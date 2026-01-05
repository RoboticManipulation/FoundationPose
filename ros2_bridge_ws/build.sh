#!/bin/bash
# Convenient build script that automatically fixes shebangs after building

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "foundationpose_ros" ]]; then
    echo "ERROR: Please activate the 'foundationpose_ros' conda environment first!"
    echo "Run: conda activate foundationpose_ros"
    exit 1
fi

echo "Building ROS2 workspace..."
colcon build --symlink-install

if [ $? -eq 0 ]; then
    echo "Build successful! Fixing shebangs..."
    ./fix_shebangs.sh
    echo ""
    echo "All done! You can now run:"
    echo "  source install/local_setup.bash"
    echo "  ros2 launch foundationpose_bridge foundationpose_service.launch.py"
else
    echo "Build failed!"
    exit 1
fi
