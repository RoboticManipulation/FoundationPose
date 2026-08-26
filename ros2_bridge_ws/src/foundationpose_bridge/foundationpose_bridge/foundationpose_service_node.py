#!/usr/bin/env python3

import numpy as np
import cv2
import trimesh
import threading
import torch
import os
import sys
import tempfile
from pathlib import Path

# Add FoundationPose to path
def _resolve_foundationpose_root():
    candidates = []
    configured_root = os.environ.get('FOUNDATIONPOSE_ROOT')
    if configured_root:
        candidates.append(Path(configured_root).expanduser())

    source_path = Path(__file__).resolve()
    candidates.extend(source_path.parents)

    for candidate in candidates:
        if (candidate / 'estimater.py').is_file():
            return candidate

    raise RuntimeError(
        "Could not locate the FoundationPose repository. Set "
        "FOUNDATIONPOSE_ROOT to the directory containing estimater.py."
    )


foundationpose_root = _resolve_foundationpose_root()
sys.path.insert(0, str(foundationpose_root))

# Import FoundationPose modules
from estimater import *
from datareader import *

# Import ROS2 modules
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# TF2 imports for publishing pose as transform
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

# Import custom service
from foundationpose_msgs.srv import LoadMesh


RESULT_MODE_CONTINUOUS_TF = 'continuous_tf'
RESULT_MODE_SERVICE_RESPONSE_ONCE = 'service_response_once'
RESULT_MODE_BOTH = 'both'
VALID_RESULT_MODES = {
    RESULT_MODE_CONTINUOUS_TF,
    RESULT_MODE_SERVICE_RESPONSE_ONCE,
    RESULT_MODE_BOTH,
}


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
        self.declare_parameter('default_result_mode', RESULT_MODE_CONTINUOUS_TF)

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
        self.default_result_mode = self._normalize_result_mode(
            self.get_parameter('default_result_mode').value,
            RESULT_MODE_CONTINUOUS_TF,
        )

        debug_path = Path(self.debug_dir).expanduser()
        if not debug_path.is_absolute():
            debug_path = foundationpose_root / debug_path
        self.debug_dir = str(debug_path)

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
        self.input_format_logged = False
        self.latest_frame = None
        self.latest_color_msg = None
        self.latest_depth_msg = None
        self.sync_slop_ns = 300_000_000

        # Control flag
        self.tracking_enabled = False

        # Threading lock for pose updates
        self.lock = threading.Lock()

        # TF2 broadcaster for publishing pose
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Match camera / RViz image display defaults (RELIABLE breaks RViz Best Effort subs)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publisher for visualization image
        self.vis_pub = self.create_publisher(ROSImage, '/FP_result', image_qos)

        # Create service
        self.srv = self.create_service(
            LoadMesh,
            'load_mesh',
            self.load_mesh_callback
        )

        # Subscribe to camera info to get intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        # Cache and pair the latest color/depth messages by timestamp. Using
        # direct rclpy subscriptions keeps the one-shot frame cache reliable
        # across the ROS 2 Humble message_filters versions used by this image.
        self.color_sub = self.create_subscription(
            ROSImage,
            self.color_topic,
            self.color_image_callback,
            image_qos,
        )
        self.depth_sub = self.create_subscription(
            ROSImage,
            self.depth_topic,
            self.depth_image_callback,
            image_qos,
        )

        self.get_logger().info("FoundationPose Service Node initialized")
        self.get_logger().info(f"Service 'load_mesh' is ready")
        self.get_logger().info(f"TF publishing: {self.camera_frame} -> {self.object_frame}")
        self.get_logger().info(f"Default result mode (when request.result_mode is empty): {self.default_result_mode}")
        self.get_logger().info(f"Score threshold: {self.score_threshold:.2f}")
        self.get_logger().info(f"Visualization: {'enabled' if self.enable_visualization else 'disabled'}")
        if self.enable_visualization:
            self.get_logger().info("Press 'q' to quit, 's' to save current frame, 'r' to reset tracking")
        self.get_logger().info("Waiting for camera intrinsics and mesh data...")

    def _normalize_result_mode(self, value, fallback):
        mode = str(value).strip().lower() if value else fallback
        if mode not in VALID_RESULT_MODES:
            self.get_logger().warn(
                f"Invalid result_mode '{value}', falling back to '{fallback}'. "
                f"Valid values: {sorted(VALID_RESULT_MODES)}"
            )
            return fallback
        return mode

    def _resolve_result_mode(self, request):
        requested = getattr(request, 'result_mode', '') or ''
        if requested.strip():
            return self._normalize_result_mode(requested, self.default_result_mode)
        return self.default_result_mode

    def _publishes_continuous_tf(self, result_mode):
        return result_mode in (RESULT_MODE_CONTINUOUS_TF, RESULT_MODE_BOTH)

    def _returns_pose_in_response(self, result_mode):
        return result_mode in (RESULT_MODE_SERVICE_RESPONSE_ONCE, RESULT_MODE_BOTH)

    def _reset_pose_state(self):
        self.pose = None
        self.frame_count = 0
        self.score_logit = None

    def _load_mesh_from_request(self, request, response):
        if len(request.data) == 0:
            response.success = False
            response.message = "Mesh data is required when enable_tracking=true"
            return False

        if request.size_bytes != len(request.data):
            self.get_logger().warn(
                f"Size mismatch: expected {request.size_bytes}, got {len(request.data)}"
            )

        temp_dir = tempfile.mkdtemp(prefix='foundationpose_mesh_')
        self.mesh_file_path = os.path.join(temp_dir, request.filename)

        with open(self.mesh_file_path, 'wb') as f:
            f.write(bytes(request.data))

        self.get_logger().info(f"Saved mesh to {self.mesh_file_path} ({len(request.data)} bytes)")

        try:
            self.mesh = trimesh.load(self.mesh_file_path)
            self.get_logger().info(f"Loaded mesh: {len(self.mesh.vertices)} vertices")
        except Exception as e:
            response.success = False
            response.message = f"Failed to load mesh: {str(e)}"
            return False

        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        return True

    def _initialize_foundationpose(self):
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

    def load_mesh_callback(self, request, response):
        """Handle mesh loading service request"""
        result_mode = self._resolve_result_mode(request)
        self.get_logger().info(
            f"Received mesh service request: enable_tracking={request.enable_tracking}, "
            f"result_mode={result_mode}"
        )
        response.pose_valid = False

        try:
            with self.lock:
                if request.enable_tracking:
                    if not self._load_mesh_from_request(request, response):
                        return response

                    self._initialize_foundationpose()
                    self._reset_pose_state()

                    pose_matrix = None
                    if self._returns_pose_in_response(result_mode):
                        if self.latest_frame is None:
                            response.success = False
                            response.message = (
                                "No synchronized RGB-D frame is cached yet; wait for camera topics before requesting "
                                "service_response_once or both"
                            )
                            return response

                        pose_matrix = self._estimate_pose_from_frame(self.latest_frame, allow_tracking=False)
                        if pose_matrix is None:
                            response.success = False
                            response.message = "Mesh loaded but one-shot pose estimation failed"
                            return response

                        response.pose = self.pose_matrix_to_transform(pose_matrix, self.latest_frame['stamp'])
                        response.pose_valid = True
                        if self._publishes_continuous_tf(result_mode):
                            self.publish_pose_tf(pose_matrix, self.latest_frame['stamp'])
                        if self.enable_visualization:
                            self.publish_visualization(pose_matrix, self.latest_frame)

                    self.tracking_enabled = self._publishes_continuous_tf(result_mode)
                    response.success = True
                    if self._returns_pose_in_response(result_mode) and self._publishes_continuous_tf(result_mode):
                        response.message = f"Mesh loaded, one-shot pose returned, and tracking enabled: {request.filename}"
                    elif self._returns_pose_in_response(result_mode):
                        response.message = f"Mesh loaded and one-shot pose returned: {request.filename}"
                    else:
                        response.message = f"Mesh loaded and tracking enabled: {request.filename}"
                    self.get_logger().info(response.message)

                else:
                    # Disable tracking
                    self.tracking_enabled = False
                    self._reset_pose_state()

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

    @staticmethod
    def _stamp_nanoseconds(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def color_image_callback(self, msg):
        self.latest_color_msg = msg
        self._process_synchronized_images()

    def depth_image_callback(self, msg):
        self.latest_depth_msg = msg
        self._process_synchronized_images()

    def _process_synchronized_images(self):
        if self.latest_color_msg is None or self.latest_depth_msg is None:
            return

        color_stamp = self._stamp_nanoseconds(self.latest_color_msg)
        depth_stamp = self._stamp_nanoseconds(self.latest_depth_msg)
        if abs(color_stamp - depth_stamp) > self.sync_slop_ns:
            return

        color_msg = self.latest_color_msg
        depth_msg = self.latest_depth_msg
        self.latest_color_msg = None
        self.latest_depth_msg = None
        self.image_callback(color_msg, depth_msg)

    def pose_matrix_to_transform(self, pose_matrix, timestamp):
        """Convert a 4x4 object-in-camera pose matrix to a TF message."""
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

        return t

    def publish_pose_tf(self, pose_matrix, timestamp):
        """Publish the pose as a TF2 transform."""
        t = self.pose_matrix_to_transform(pose_matrix, timestamp)

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

    def prepare_frame(self, color_msg, depth_msg):
        """Convert synchronized ROS image messages into FoundationPose inputs."""
        if not self.intrinsics_received:
            self.get_logger().warn("Camera intrinsics not yet received, skipping frame", throttle_duration_sec=5.0)
            return None

        try:
            # Convert ROS Image messages to OpenCV/numpy format
            if color_msg.encoding == 'rgb8':
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
            elif color_msg.encoding == 'bgr8':
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            elif color_msg.encoding == 'rgba8':
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgba8')
                color = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
            elif color_msg.encoding == 'bgra8':
                color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgra8')
                color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
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

            # Align depth to color resolution (required when camera streams differ in size)
            color_h, color_w = color.shape[:2]
            depth_h, depth_w = depth.shape[:2]
            if (depth_h, depth_w) != (color_h, color_w):
                if not self.input_format_logged:
                    self.get_logger().warn(
                        f"Depth {depth_w}x{depth_h} != color {color_w}x{color_h}, resizing depth to match color"
                    )
                depth = cv2.resize(depth, (color_w, color_h), interpolation=cv2.INTER_NEAREST)

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
            if not self.input_format_logged:
                self.get_logger().info(f"Color shape: {color.shape}, dtype: {color.dtype}")
                self.get_logger().info(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
                self.get_logger().info(f"Depth range: min={depth.min():.3f}m, max={depth.max():.3f}m")
                self.input_format_logged = True

        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return None

        return {
            'color': color,
            'depth': depth,
            'K': K_scaled,
            'stamp': color_msg.header.stamp,
        }

    def _estimate_pose_from_frame(self, frame, allow_tracking):
        color = frame['color']
        depth = frame['depth']
        K_scaled = frame['K']

        # Pose estimation
        if self.pose is None or not allow_tracking:
            # Initial pose estimation
            mask = np.ones(color.shape[:2], dtype=np.uint8) * 255

            # # Check valid depth
            # valid_depth_in_mask = np.sum((depth > 0.1) & (depth < 3.0) & (mask > 0))
            # self.get_logger().info(f"Valid depth points in mask: {valid_depth_in_mask}")

            # if valid_depth_in_mask < 1000:
            #     self.get_logger().warn(f"Not enough valid depth points ({valid_depth_in_mask}), skipping frame")
            #     return None

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
                return None

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
                return None

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
                return None

        # Preserve FoundationPose's canonical mesh frame for service responses
        # and TF. The oriented-bounding-box frame is only for visualization.
        return self.pose

    def publish_visualization(self, pose_matrix, frame):
        color = frame['color']
        K_scaled = frame['K']
        center_pose = pose_matrix @ np.linalg.inv(self.to_origin)
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

        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            vis_msg.header.stamp = frame['stamp']
            vis_msg.header.frame_id = self.camera_frame
            self.vis_pub.publish(vis_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish visualization: {e}")

    def image_callback(self, color_msg, depth_msg):
        """Process synchronized color and depth images."""
        frame = self.prepare_frame(color_msg, depth_msg)
        if frame is None:
            return

        with self.lock:
            self.latest_frame = frame
            if not self.tracking_enabled:
                return

            if self.est is None:
                self.get_logger().warn("FoundationPose estimator not initialized, skipping frame", throttle_duration_sec=5.0)
                return

            pose_matrix = self._estimate_pose_from_frame(frame, allow_tracking=True)
            if pose_matrix is None:
                return

            self.publish_pose_tf(pose_matrix, frame['stamp'])
            if self.enable_visualization:
                self.publish_visualization(pose_matrix, frame)
            self.frame_count += 1


def main(args=None):
    rclpy.init(args=args)

    node = FoundationPoseServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down")
    finally:
        # if node.enable_visualization:
        #     cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node stopped")


if __name__ == '__main__':
    main()
