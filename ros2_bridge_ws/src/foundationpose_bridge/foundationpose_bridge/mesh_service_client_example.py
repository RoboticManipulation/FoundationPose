#!/usr/bin/env python3

"""
Example client for the FoundationPose mesh loading service.

This script demonstrates how to:
1. Load a mesh file and enable tracking
2. Disable tracking
"""

import os
import rclpy
from rclpy.node import Node
from foundationpose_msgs.srv import LoadMesh
import sys


class MeshServiceClient(Node):
    def __init__(self):
        super().__init__('mesh_service_client')
        self.client = self.create_client(LoadMesh, 'load_mesh')

        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service "load_mesh" to become available...')

    def log_pose_response(self, response):
        """Print one-shot pose output when the service returns one."""
        if not response.pose_valid:
            return

        pose = response.pose.transform
        self.get_logger().info(
            'One-shot pose received: '
            f'frame={response.pose.header.frame_id}, '
            f'child={response.pose.child_frame_id}, '
            f'translation=({pose.translation.x:.4f}, '
            f'{pose.translation.y:.4f}, {pose.translation.z:.4f}), '
            f'rotation=({pose.rotation.x:.4f}, {pose.rotation.y:.4f}, '
            f'{pose.rotation.z:.4f}, {pose.rotation.w:.4f}), '
            f'score={response.score:.2f}'
        )

    def load_mesh_and_enable_tracking(self, mesh_file_path, result_mode='both'):
        """Load a mesh file and enable tracking."""
        self.get_logger().info(f'Loading mesh from {mesh_file_path}...')

        # Read mesh file
        try:
            with open(mesh_file_path, 'rb') as f:
                mesh_data = f.read()
        except FileNotFoundError:
            self.get_logger().error(f'Mesh file not found: {mesh_file_path}')
            return False

        # Create service request
        request = LoadMesh.Request()
        request.filename = os.path.basename(mesh_file_path)
        request.data = list(mesh_data)  # Convert bytes to list of uint8
        request.size_bytes = len(mesh_data)
        request.enable_tracking = True
        request.result_mode = result_mode
        request.has_registration_mask = False

        self.get_logger().info(
            f'Sending request: {request.filename} ({request.size_bytes} bytes), '
            f'enable_tracking=True, result_mode={result_mode}'
        )

        # Call service
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Success: {response.message}')
                self.log_pose_response(response)
                return True
            else:
                self.get_logger().error(f'Failed: {response.message}')
                return False
        else:
            self.get_logger().error('Service call failed')
            return False

    def disable_tracking(self):
        """Disable tracking."""
        self.get_logger().info('Disabling tracking...')

        # Create service request with empty mesh data
        request = LoadMesh.Request()
        request.filename = ''
        request.data = []
        request.size_bytes = 0
        request.enable_tracking = False
        request.result_mode = ''
        request.has_registration_mask = False

        # Call service
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Success: {response.message}')
                return True
            else:
                self.get_logger().error(f'Failed: {response.message}')
                return False
        else:
            self.get_logger().error('Service call failed')
            return False


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  Enable tracking: ros2 run foundationpose_bridge "
            "mesh_service_client_example <mesh_file_path> [result_mode]"
        )
        print(
            "  Disable tracking: ros2 run foundationpose_bridge "
            "mesh_service_client_example disable"
        )
        print(
            "  result_mode: continuous_tf | service_response_once | both "
            "(default: both)"
        )
        sys.exit(1)

    client = MeshServiceClient()

    if sys.argv[1] == 'disable':
        # Disable tracking
        client.disable_tracking()
    else:
        # Load mesh and enable tracking
        mesh_file_path = sys.argv[1]
        result_mode = sys.argv[2] if len(sys.argv) > 2 else 'both'
        client.load_mesh_and_enable_tracking(mesh_file_path, result_mode=result_mode)

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
