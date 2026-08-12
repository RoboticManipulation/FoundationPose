#!/usr/bin/env python3

import numpy as np
import cv2
import trimesh
import threading
import torch
import os
import sys
import tempfile

# Add FoundationPose to path
# Use absolute path since relative path differs between source and install
foundationpose_root = '/home/ehsanullahm1/ros2/object_placement/FoundationPose'
if os.path.exists(foundationpose_root):
    sys.path.insert(0, foundationpose_root)
else:
    print(f"WARNING: FoundationPose root not found at {foundationpose_root}")
    # Try relative path as fallback
    foundationpose_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    sys.path.insert(0, foundationpose_root)

# Import FoundationPose modules
from estimater import *
from datareader import *

# Import ROS2 modules
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# TF2 imports for publishing pose as transform
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

# Import custom service
from foundationpose_msgs.srv import LoadMesh


def _to_scalar(x) -> float:
    """Convert various types (torch.Tensor, numpy array) to a scalar float."""
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().view(-1)[0].item()
    except Exception:
        pass
    try:
        arr = np.asarray(x).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        pass
    return None


class FoundationPoseServiceNode(Node):
    def __init__(self):
        super().__init__('foundationpose_service_node')

        # Declare parameters
        self.declare_parameter('debug', 1)
        self.declare_parameter('debug_dir', 'debug_ros2_service')
        self.declare_parameter('downscale', 1.0)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_frame', 'foundationpose_object')
        self.declare_parameter('score_threshold', 100.0)
        self.declare_parameter('color_topic', '/sim_camera_rgb')
        self.declare_parameter('depth_topic', '/sim_camera_depth')
        self.declare_parameter('camera_info_topic', '/sim_camera_info')
        self.declare_parameter('enable_visualization', True)

        # Get parameters
        self.debug = self.get_parameter('debug').value
        self.debug_dir = self.get_parameter('debug_dir').value
        self.downscale = self.get_parameter('downscale').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.object_frame = self.get_parameter('object_frame').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.enable_visualization = self.get_parameter('enable_visualization').value

        self.bridge = CvBridge()

        # Setup debug directory
        os.makedirs(self.debug_dir, exist_ok=True)

        # Camera intrinsics (will be set from camera_info)
        self.K = None
        self.D = None
        self.intrinsics_received = False

        # FoundationPose state
        self.pose = None
        self.frame_count = 0
        self.est = None
        self.to_origin = None
        self.bbox = None
        self.score_logit = None
        self.mesh = None
        self.mesh_file_path = None

        # Control flag
        self.tracking_enabled = False

        # Threading lock for pose updates
        self.lock = threading.Lock()

        # TF2 broadcaster for publishing pose
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create service
        self.srv = self.create_service(
            LoadMesh,
            'load_mesh',
            self.load_mesh_callback
        )

        # QoS profile to match publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to camera info to get intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        # Synchronized subscribers for color and depth images
        self.color_sub = message_filters.Subscriber(
            self,
            ROSImage,
            self.color_topic,
            qos_profile=qos
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            ROSImage,
            self.depth_topic,
            qos_profile=qos
        )

        # Time synchronizer for color and depth
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=20,
            slop=0.3
        )
        self.sync.registerCallback(self.image_callback)

        self.get_logger().info("FoundationPose Service Node initialized")
        self.get_logger().info(f"Service 'load_mesh' is ready")
        self.get_logger().info(f"TF publishing: {self.camera_frame} -> {self.object_frame}")
        self.get_logger().info(f"Score threshold: {self.score_threshold:.2f}")
        self.get_logger().info(f"Visualization: {'enabled' if self.enable_visualization else 'disabled'}")
        if self.enable_visualization:
            self.get_logger().info("Press 'q' to quit, 's' to save current frame, 'r' to reset tracking")
        self.get_logger().info("Waiting for camera intrinsics and mesh data...")

    def load_mesh_callback(self, request, response):
        """Handle mesh loading service request"""
        self.get_logger().info(f"Received mesh service request: enable_tracking={request.enable_tracking}")

        try:
            with self.lock:
                if request.enable_tracking:
                    # Enable tracking - mesh is required
                    if len(request.data) == 0:
                        response.success = False
                        response.message = "Mesh data is required when enable_tracking=true"
                        return response

                    # Verify size
                    if request.size_bytes != len(request.data):
                        self.get_logger().warn(
                            f"Size mismatch: expected {request.size_bytes}, got {len(request.data)}"
                        )

                    # Save mesh data to temporary file
                    temp_dir = tempfile.mkdtemp(prefix='foundationpose_mesh_')
                    self.mesh_file_path = os.path.join(temp_dir, request.filename)

                    with open(self.mesh_file_path, 'wb') as f:
                        f.write(bytes(request.data))

                    self.get_logger().info(f"Saved mesh to {self.mesh_file_path} ({len(request.data)} bytes)")

                    # Load mesh
                    try:
                        self.mesh = trimesh.load(self.mesh_file_path)
                        self.get_logger().info(f"Loaded mesh: {len(self.mesh.vertices)} vertices")
                    except Exception as e:
                        response.success = False
                        response.message = f"Failed to load mesh: {str(e)}"
                        return response

                    # Compute oriented bounding box
                    self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
                    self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

                    # Initialize FoundationPose
                    self.get_logger().info("Initializing FoundationPose estimator...")
                    scorer = ScorePredictor()
                    refiner = PoseRefinePredictor()
                    glctx = dr.RasterizeCudaContext()

                    self.est = FoundationPose(
                        model_pts=self.mesh.vertices,
                        model_normals=self.mesh.vertex_normals,
                        mesh=self.mesh,
                        scorer=scorer,
                        refiner=refiner,
                        debug_dir=self.debug_dir,
                        debug=self.debug,
                        glctx=glctx
                    )

                    # Reset pose estimation
                    self.pose = None
                    self.frame_count = 0
                    self.score_logit = None

                    # Enable tracking
                    self.tracking_enabled = True

                    response.success = True
                    response.message = f"Mesh loaded and tracking enabled: {request.filename}"
                    self.get_logger().info(response.message)

                else:
                    # Disable tracking
                    self.tracking_enabled = False
                    self.pose = None

                    response.success = True
                    response.message = "Tracking disabled"
                    self.get_logger().info(response.message)

        except Exception as e:
            response.success = False
            response.message = f"Service error: {str(e)}"
            self.get_logger().error(response.message)

        return response

    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo message"""
        if not self.intrinsics_received:
            self.get_logger().info(f"Image resolution from CameraInfo: {msg.width}x{msg.height}")

            # Extract K matrix (intrinsic camera matrix)
            k_mat = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"K matrix (intrinsic):\n{k_mat}")

            self.K = k_mat
            self.intrinsics_received = True
            self.get_logger().info(f"Camera intrinsics received")

    def publish_pose_tf(self, pose_matrix, timestamp):
        """Publish the pose as a TF2 transform

        Args:
            pose_matrix: 4x4 transformation matrix (object in camera frame)
            timestamp: ROS2 timestamp from the image message
        """
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.object_frame

        # Extract translation from 4x4 matrix
        t.transform.translation.x = float(pose_matrix[0, 3])
        t.transform.translation.y = float(pose_matrix[1, 3])
        t.transform.translation.z = float(pose_matrix[2, 3])

        # Extract rotation matrix and convert to quaternion
        rotation_matrix = pose_matrix[:3, :3]
        quat = R.from_matrix(rotation_matrix).as_quat()  # Returns [x, y, z, w]

        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

    def image_callback(self, color_msg, depth_msg):
        """Process synchronized color and depth images"""
        # Skip if tracking is not enabled
        if not self.tracking_enabled:
            return

        if not self.intrinsics_received:
            self.get_logger().warn("Camera intrinsics not yet received, skipping frame", throttle_duration_sec=5.0)
            return

        if self.est is None:
            self.get_logger().warn("FoundationPose estimator not initialized, skipping frame", throttle_duration_sec=5.0)
            return

        try:
            # Convert ROS Image messages to OpenCV/numpy format
            if color_msg.encoding == 'rgb8':
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            elif color_msg.encoding == 'bgr8':
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            elif color_msg.encoding == 'rgba8':
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgra8')
                color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            elif color_msg.encoding == 'bgra8':
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgra8')
                color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            else:
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')
                if len(color.shape) == 2:
                    color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)

            # Ensure color is contiguous
            color = np.ascontiguousarray(color)

            # Depth image handling
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Ensure depth is 2D
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            depth = np.squeeze(depth)
            depth = np.ascontiguousarray(depth)

            # Handle depth conversion based on encoding
            if depth_msg.encoding == '16UC1':
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)

            # Handle invalid depth values
            depth = np.where(np.isinf(depth) | np.isnan(depth), 0.0, depth)

            # Downscale images if requested
            scale = self.downscale
            K_scaled = self.K.copy()
            if scale != 1.0:
                new_h, new_w = int(color.shape[0] * scale), int(color.shape[1] * scale)
                color = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                K_scaled[0, 0] *= scale  # fx
                K_scaled[1, 1] *= scale  # fy
                K_scaled[0, 2] *= scale  # cx
                K_scaled[1, 2] *= scale  # cy

            # Debug: Log once to verify formats
            if self.frame_count == 0:
                self.get_logger().info(f"Color shape: {color.shape}, dtype: {color.dtype}")
                self.get_logger().info(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
                self.get_logger().info(f"Depth range: min={depth.min():.3f}m, max={depth.max():.3f}m")

        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        with self.lock:
            # Pose estimation
            if self.pose is None:
                # Initial pose estimation
                mask = np.ones(color.shape[:2], dtype=np.uint8) * 255

                # Check valid depth
                valid_depth_in_mask = np.sum((depth > 0.1) & (depth < 3.0) & (mask > 0))
                self.get_logger().info(f"Valid depth points in mask: {valid_depth_in_mask}")

                if valid_depth_in_mask < 1000:
                    self.get_logger().warn(f"Not enough valid depth points ({valid_depth_in_mask}), skipping frame")
                    return

                self.get_logger().info("Running initial pose estimation...")
                try:
                    torch.cuda.empty_cache()
                    self.pose = self.est.register(K=K_scaled, rgb=color, depth=depth, ob_mask=mask, iteration=3)
                except Exception as e:
                    self.get_logger().error(f"Registration failed: {e}")
                    torch.cuda.empty_cache()
                    self.pose = None

                if self.pose is None:
                    self.get_logger().warn("Failed to estimate initial pose, skipping frame")
                    return

                self.get_logger().info("Initial pose registration successful!")

                # Extract score
                try:
                    if hasattr(self.est, 'scores') and self.est.scores is not None:
                        self.score_logit = _to_scalar(self.est.scores[0])
                except Exception as e:
                    self.get_logger().info(f"Failed to read init score: {e}")
            else:
                # Track with refiner
                try:
                    self.pose = self.est.track_one(rgb=color, depth=depth, K=K_scaled, iteration=2)
                except RuntimeError as e:
                    self.get_logger().warn(f"Tracking failed: {e}, resetting pose...")
                    self.pose = None
                    return

                # Compute score for tracking frame
                try:
                    cur_pose_centered = getattr(self.est, 'pose_last', None)
                    if cur_pose_centered is not None:
                        scores, _ = self.est.scorer.predict(
                            mesh=self.est.mesh,
                            rgb=color,
                            depth=depth,
                            K=K_scaled,
                            ob_in_cams=cur_pose_centered.data.cpu().numpy().reshape(1, 4, 4),
                            normal_map=None,
                            mesh_tensors=self.est.mesh_tensors,
                            glctx=self.est.glctx,
                            mesh_diameter=self.est.diameter,
                            get_vis=False,
                        )
                        self.score_logit = _to_scalar(scores)
                except Exception as e:
                    self.get_logger().info(f"Failed to compute score on track frame: {e}")

                # Check score threshold
                if self.score_logit is not None and self.score_logit > self.score_threshold:
                    self.get_logger().warn(f"Score {self.score_logit:.2f} above threshold, resetting tracking...")
                    self.pose = None
                    self.score_logit = None
                    return

            # Visualize
            center_pose = self.pose @ np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(K_scaled, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K_scaled, thickness=3, transparency=0, is_input_rgb=True)

            # Display score
            if self.score_logit is not None:
                try:
                    cv2.putText(
                        vis,
                        f"score_logit: {self.score_logit:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                except Exception:
                    pass

            # Publish the canonical mesh pose returned by FoundationPose.
            # ``center_pose`` is the oriented-bounding-box frame used only for
            # drawing; publishing it silently permutes/rotates object axes.
            self.publish_pose_tf(self.pose, color_msg.header.stamp)

            # Visualization (optional)
            if self.enable_visualization:
                cv2.imshow('FoundationPose Service Node', vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Quit requested")
                    rclpy.shutdown()
                elif key == ord('r'):
                    self.get_logger().info("Resetting tracking...")
                    self.pose = None
                elif key == ord('s'):
                    save_path = f"{self.debug_dir}/frame_{self.frame_count:04d}.png"
                    cv2.imwrite(save_path, vis)
                    self.get_logger().info(f"Saved frame to {save_path}")

            self.frame_count += 1


def main(args=None):
    rclpy.init(args=args)

    node = FoundationPoseServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down")
    finally:
        if node.enable_visualization:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node stopped")


if __name__ == '__main__':
    main()
