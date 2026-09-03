#!/usr/bin/env python3

import os
import sys
import tempfile
import threading
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from foundationpose_msgs.srv import LoadMesh
from geometry_msgs.msg import TransformStamped
import message_filters
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage
import tf2_ros
import torch
import trimesh


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
import nvdiffrast.torch as dr  # noqa: E402
from estimater import FoundationPose  # noqa: E402
from learning.training.predict_pose_refine import (  # noqa: E402
    PoseRefinePredictor,
)
from learning.training.predict_score import ScorePredictor  # noqa: E402
from Utils import draw_posed_3d_box, draw_xyz_axis  # noqa: E402


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
        # FoundationPose ranks candidate poses by descending score, so this is
        # a minimum acceptable tracking score (not a maximum).
        self.declare_parameter('score_threshold', 40.0)
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
        self.registration_mask = None
        self.registration_mask_logged = False

        # Control flag
        self.tracking_enabled = False

        # Threading lock for pose updates
        self.lock = threading.Lock()

        # TF2 broadcaster for publishing pose
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Match the Orbbec camera's advertised QoS. Using a synchronized
        # message-filter pair also prevents one high-rate stream from starving
        # the other in the single-threaded executor.
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publish the generated image reliably so the default RViz Image
        # display (Reliable) can subscribe to it.
        visualization_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publisher for visualization image
        self.vis_pub = self.create_publisher(
            ROSImage,
            '/FP_result',
            visualization_qos,
        )

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

        # Synchronize color and depth before conversion. image_callback caches
        # the converted pair even when continuous tracking is disabled, which
        # keeps service_response_once requests supported.
        self.color_sub = message_filters.Subscriber(
            self,
            ROSImage,
            self.color_topic,
            qos_profile=image_qos,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            ROSImage,
            self.depth_topic,
            qos_profile=image_qos,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=20,
            slop=0.3,
        )
        self.sync.registerCallback(self.image_callback)

        self.get_logger().info("FoundationPose Service Node initialized")
        self.get_logger().info("Service 'load_mesh' is ready")
        self.get_logger().info(
            f"TF publishing: {self.camera_frame} -> {self.object_frame}"
        )
        self.get_logger().info(
            "Default result mode (when request.result_mode is empty): "
            f"{self.default_result_mode}"
        )
        self.get_logger().info(f"Minimum tracking score: {self.score_threshold:.2f}")
        visualization_state = 'enabled' if self.enable_visualization else 'disabled'
        self.get_logger().info(f"Visualization: {visualization_state}")
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

    def _reset_registration_mask(self):
        self.registration_mask = None
        self.registration_mask_logged = False

    def _begin_enable_request(self):
        """Stop any previous run before replacing estimator inputs."""
        self.tracking_enabled = False
        self._reset_pose_state()
        self._reset_registration_mask()
        self.est = None
        self.mesh = None
        self.mesh_file_path = None
        self.to_origin = None
        self.bbox = None

    def _validate_estimate(self, context):
        """Reject malformed or unscored estimates before publishing them."""
        pose = np.asarray(self.pose)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            self.get_logger().warn(
                f"{context} produced an invalid 4x4 pose; rejecting it"
            )
            self._reset_pose_state()
            return False

        if self.score_logit is None or not np.isfinite(self.score_logit):
            self.get_logger().warn(
                f"{context} did not produce a finite score; rejecting the pose"
            )
            self._reset_pose_state()
            return False

        return True

    def _load_registration_mask_from_request(self, request, response):
        """Validate and cache an optional object mask from a service request."""
        self._reset_registration_mask()

        if not getattr(request, 'has_registration_mask', False):
            self.get_logger().warn(
                "No registration mask was supplied; falling back to the full image. "
                "Pose quality may be reduced."
            )
            return True

        mask_msg = getattr(request, 'registration_mask', None)
        if (
            mask_msg is None
            or mask_msg.width <= 0
            or mask_msg.height <= 0
            or len(mask_msg.data) == 0
        ):
            response.success = False
            response.message = (
                "has_registration_mask=true, but registration_mask is empty"
            )
            return False

        encoding = str(mask_msg.encoding).strip().lower()
        if encoding != 'mono8':
            self.get_logger().warn(
                f"Registration mask encoding is '{mask_msg.encoding}', converting to mono8"
            )

        mask_frame = str(mask_msg.header.frame_id).strip().lstrip('/')
        expected_frame = str(self.camera_frame).strip().lstrip('/')
        if mask_frame and mask_frame != expected_frame:
            response.success = False
            response.message = (
                f"registration_mask frame '{mask_frame}' does not match "
                f"camera frame '{expected_frame}'"
            )
            return False

        try:
            mask = self.bridge.imgmsg_to_cv2(
                mask_msg,
                desired_encoding='mono8',
            )
        except Exception as e:
            response.success = False
            response.message = f"Failed to convert registration mask to mono8: {e}"
            return False

        mask = np.asarray(mask)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2 or mask.size == 0:
            response.success = False
            response.message = (
                "registration_mask must convert to a nonempty single-channel image"
            )
            return False
        expected_shape = (int(mask_msg.height), int(mask_msg.width))
        if mask.shape != expected_shape:
            response.success = False
            response.message = (
                f"registration_mask dimensions do not match its metadata: "
                f"decoded={mask.shape[1]}x{mask.shape[0]}, "
                f"declared={mask_msg.width}x{mask_msg.height}"
            )
            return False

        mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
        foreground_pixels = int(np.count_nonzero(mask))
        if foreground_pixels == 0:
            response.success = False
            response.message = "registration_mask contains no foreground pixels"
            return False

        self.registration_mask = mask
        self.get_logger().info(
            f"Received registration mask: {mask.shape[1]}x{mask.shape[0]}, "
            f"foreground_pixels={foreground_pixels}"
        )
        return True

    def _registration_mask_for_frame(self, frame_shape):
        """Return the cached binary mask at the prepared color-frame size."""
        frame_h, frame_w = frame_shape
        if self.registration_mask is None:
            return np.ones((frame_h, frame_w), dtype=np.uint8) * 255

        mask = self.registration_mask
        source_h, source_w = mask.shape
        if (source_h, source_w) != (frame_h, frame_w):
            if source_w * frame_h != frame_w * source_h:
                self.get_logger().error(
                    f"Registration mask aspect ratio {source_w}x{source_h} "
                    f"does not match prepared frame {frame_w}x{frame_h}"
                )
                return None
            mask = cv2.resize(
                mask,
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            )

        mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
        foreground_pixels = int(np.count_nonzero(mask))
        if foreground_pixels == 0:
            self.get_logger().error(
                "Registration mask has no foreground pixels after resizing to the prepared frame"
            )
            return None

        if not self.registration_mask_logged:
            self.get_logger().info(
                f"Using registration mask: source={source_w}x{source_h}, "
                f"prepared_frame={frame_w}x{frame_h}, "
                f"foreground_pixels={foreground_pixels}"
            )
            self.registration_mask_logged = True

        return mask

    def _load_mesh_from_request(self, request, response):
        if len(request.data) == 0:
            response.success = False
            response.message = "Mesh data is required when enable_tracking=true"
            return False

        if request.size_bytes != len(request.data):
            response.success = False
            response.message = (
                f"Mesh size mismatch: expected {request.size_bytes}, "
                f"received {len(request.data)}"
            )
            return False

        filename = str(request.filename).strip()
        if (
            not filename
            or filename in {'.', '..'}
            or Path(filename).name != filename
            or '/' in filename
            or '\\' in filename
        ):
            response.success = False
            response.message = "Mesh filename must be a nonempty basename"
            return False

        temp_dir = tempfile.mkdtemp(prefix='foundationpose_mesh_')
        mesh_file_path = os.path.join(temp_dir, filename)

        with open(mesh_file_path, 'wb') as f:
            f.write(bytes(request.data))

        self.get_logger().info(
            f"Saved mesh to {mesh_file_path} ({len(request.data)} bytes)"
        )

        try:
            mesh = trimesh.load(mesh_file_path, force='mesh')
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            if (
                not isinstance(mesh, trimesh.Trimesh)
                or mesh.is_empty
                or vertices.ndim != 2
                or vertices.shape[0] == 0
                or faces.ndim != 2
                or faces.shape[0] == 0
                or not np.all(np.isfinite(vertices))
            ):
                raise ValueError("mesh must contain finite vertices and faces")

            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            if not np.all(np.isfinite(to_origin)) or not np.all(np.isfinite(extents)):
                raise ValueError("mesh oriented bounds are not finite")
        except Exception as e:
            response.success = False
            response.message = f"Failed to load mesh: {str(e)}"
            return False

        self.mesh_file_path = mesh_file_path
        self.mesh = mesh
        self.to_origin = to_origin
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        self.get_logger().info(f"Loaded mesh: {len(self.mesh.vertices)} vertices")
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
        """Handle a mesh loading service request."""
        result_mode = self._resolve_result_mode(request)
        self.get_logger().info(
            f"Received mesh service request: enable_tracking={request.enable_tracking}, "
            f"result_mode={result_mode}"
        )
        response.pose_valid = False

        try:
            with self.lock:
                if request.enable_tracking:
                    # Every enable request replaces the active estimator. Stop
                    # the previous run first so all failure paths remain safe.
                    self._begin_enable_request()

                    if not self._load_registration_mask_from_request(
                        request,
                        response,
                    ):
                        return response

                    if (
                        self.registration_mask is not None
                        and self.latest_frame is not None
                        and self._registration_mask_for_frame(
                            self.latest_frame['color'].shape[:2]
                        ) is None
                    ):
                        response.success = False
                        response.message = (
                            "registration_mask is incompatible with the latest "
                            "synchronized RGB-D frame"
                        )
                        self._reset_registration_mask()
                        return response

                    if not self._load_mesh_from_request(request, response):
                        self._reset_registration_mask()
                        return response

                    self._initialize_foundationpose()
                    self._reset_pose_state()

                    pose_matrix = None
                    if self._returns_pose_in_response(result_mode):
                        if self.latest_frame is None:
                            response.success = False
                            response.message = (
                                "No synchronized RGB-D frame is cached yet; wait "
                                "for camera topics before requesting "
                                "service_response_once or both"
                            )
                            self._reset_registration_mask()
                            return response

                        pose_matrix = self._estimate_pose_from_frame(
                            self.latest_frame,
                            allow_tracking=False,
                        )
                        if pose_matrix is None:
                            response.success = False
                            response.message = "Mesh loaded but one-shot pose estimation failed"
                            self._reset_registration_mask()
                            return response

                        response.pose = self.pose_matrix_to_transform(
                            pose_matrix,
                            self.latest_frame['stamp'],
                        )
                        response.pose_valid = True
                        if self.score_logit is not None:
                            response.score = float(self.score_logit)
                        if self._publishes_continuous_tf(result_mode):
                            self.publish_pose_tf(pose_matrix, self.latest_frame['stamp'])
                        if self.enable_visualization:
                            self.publish_visualization(pose_matrix, self.latest_frame)

                    self.tracking_enabled = self._publishes_continuous_tf(result_mode)
                    response.success = True
                    returns_pose = self._returns_pose_in_response(result_mode)
                    publishes_tf = self._publishes_continuous_tf(result_mode)
                    if returns_pose and publishes_tf:
                        response.message = (
                            "Mesh loaded, one-shot pose returned, and tracking "
                            f"enabled: {request.filename}"
                        )
                    elif self._returns_pose_in_response(result_mode):
                        response.message = (
                            "Mesh loaded and one-shot pose returned: "
                            f"{request.filename}"
                        )
                    else:
                        response.message = f"Mesh loaded and tracking enabled: {request.filename}"
                    self.get_logger().info(response.message)

                else:
                    # Disable tracking
                    self.tracking_enabled = False
                    self._reset_pose_state()
                    self._reset_registration_mask()

                    response.success = True
                    response.message = "Tracking disabled"
                    self.get_logger().info(response.message)

        except Exception as e:
            self._reset_registration_mask()
            response.success = False
            response.message = f"Service error: {str(e)}"
            self.get_logger().error(response.message)

        return response

    def camera_info_callback(self, msg):
        """Extract camera intrinsics from a CameraInfo message."""
        if not self.intrinsics_received:
            self.get_logger().info(
                f"Image resolution from CameraInfo: {msg.width}x{msg.height}"
            )

            # Extract K matrix (intrinsic camera matrix)
            k_mat = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"K matrix (intrinsic):\n{k_mat}")

            self.K = k_mat
            self.intrinsics_received = True
            self.get_logger().info("Camera intrinsics received")

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
            self.get_logger().warn(
                "Camera intrinsics not yet received, skipping frame",
                throttle_duration_sec=5.0,
            )
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

            # Align depth when camera streams have different resolutions.
            color_h, color_w = color.shape[:2]
            depth_h, depth_w = depth.shape[:2]
            if (depth_h, depth_w) != (color_h, color_w):
                if not self.input_format_logged:
                    self.get_logger().warn(
                        f"Depth {depth_w}x{depth_h} != color "
                        f"{color_w}x{color_h}; resizing depth to match color"
                    )
                depth = cv2.resize(
                    depth,
                    (color_w, color_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Downscale images if requested
            scale = self.downscale
            K_scaled = self.K.copy()
            if scale != 1.0:
                new_h = int(color.shape[0] * scale)
                new_w = int(color.shape[1] * scale)
                color = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(
                    depth,
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                K_scaled[0, 0] *= scale  # fx
                K_scaled[1, 1] *= scale  # fy
                K_scaled[0, 2] *= scale  # cx
                K_scaled[1, 2] *= scale  # cy

            # Debug: Log once to verify formats
            if not self.input_format_logged:
                self.get_logger().info(
                    f"Color shape: {color.shape}, dtype: {color.dtype}"
                )
                self.get_logger().info(
                    f"Depth shape: {depth.shape}, dtype: {depth.dtype}"
                )
                self.get_logger().info(
                    f"Depth range: min={depth.min():.3f}m, "
                    f"max={depth.max():.3f}m"
                )
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
            mask = self._registration_mask_for_frame(color.shape[:2])
            if mask is None:
                self.get_logger().warn("Cannot register without a valid object mask")
                if allow_tracking:
                    self.tracking_enabled = False
                    self.get_logger().error(
                        "Tracking disabled because the registration mask is "
                        "incompatible with the synchronized RGB-D frame"
                    )
                return None

            # Clear estimator-side state as well as the node's cached score.
            # FoundationPose can return a translation-only fallback when fewer
            # than four masked depth pixels are valid, without updating its
            # score or last pose. Stale values from a prior successful frame
            # must never make that fallback look like a valid registration.
            self.score_logit = None
            self.est.scores = None
            self.est.pose_last = None

            valid_masked_depth = (
                (mask > 0)
                & np.isfinite(depth)
                & (depth >= 0.001)
            )
            valid_depth_pixels = int(np.count_nonzero(valid_masked_depth))
            if valid_depth_pixels < 4:
                self.get_logger().warn(
                    "Initial registration has fewer than four valid masked "
                    f"depth pixels ({valid_depth_pixels}); rejecting frame"
                )
                return None

            self.get_logger().info("Running initial pose estimation...")
            try:
                torch.cuda.empty_cache()
                self.pose = self.est.register(
                    K=K_scaled,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=3,
                )
            except Exception as e:
                self.get_logger().error(f"Registration failed: {e}")
                torch.cuda.empty_cache()
                self.pose = None

            if self.pose is None:
                self.get_logger().warn("Failed to estimate initial pose, skipping frame")
                return None

            # Extract score
            try:
                if hasattr(self.est, 'scores') and self.est.scores is not None:
                    self.score_logit = _to_scalar(self.est.scores[0])
            except Exception as e:
                self.get_logger().info(f"Failed to read init score: {e}")

            if not self._validate_estimate("Initial registration"):
                return None

            self.get_logger().info("Initial pose registration successful!")

            # Reject low-confidence registrations before returning them.
            # FoundationPose ranks candidates by descending score, so a
            # below-threshold winner means every hypothesis it considered was
            # a poor match (bad mask, occlusion, symmetry ambiguity, ...).
            # This applies to every fresh registration - the one-shot service
            # response path and the first frame of continuous/both tracking -
            # since both reach this branch with self.pose is None.
            if self.score_logit is not None and self.score_logit < self.score_threshold:
                self.get_logger().warn(
                    f"Initial registration score {self.score_logit:.2f} below "
                    f"minimum {self.score_threshold:.2f}, rejecting pose"
                )
                self.pose = None
                self.score_logit = None
                return None
        else:
            # Track with refiner
            try:
                self.pose = self.est.track_one(
                    rgb=color,
                    depth=depth,
                    K=K_scaled,
                    iteration=2,
                )
            except RuntimeError as e:
                self.get_logger().warn(f"Tracking failed: {e}, resetting pose...")
                self.pose = None
                return None

            # Compute score for tracking frame
            self.score_logit = None
            try:
                cur_pose_centered = getattr(self.est, 'pose_last', None)
                if cur_pose_centered is not None:
                    scores, _ = self.est.scorer.predict(
                        mesh=self.est.mesh,
                        rgb=color,
                        depth=depth,
                        K=K_scaled,
                        ob_in_cams=(
                            cur_pose_centered.data.cpu().numpy().reshape(1, 4, 4)
                        ),
                        normal_map=None,
                        mesh_tensors=self.est.mesh_tensors,
                        glctx=self.est.glctx,
                        mesh_diameter=self.est.diameter,
                        get_vis=False,
                    )
                    self.score_logit = _to_scalar(scores)
            except Exception as e:
                self.get_logger().info(f"Failed to compute score on track frame: {e}")

            if not self._validate_estimate("Tracking"):
                return None

            # Higher FoundationPose scores are better. Re-register only when
            # the current pose falls below the minimum acceptable score.
            if self.score_logit is not None and self.score_logit < self.score_threshold:
                self.get_logger().warn(
                    f"Score {self.score_logit:.2f} below minimum "
                    f"{self.score_threshold:.2f}, resetting tracking..."
                )
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
        vis = draw_posed_3d_box(
            K_scaled,
            img=color,
            ob_in_cam=center_pose,
            bbox=self.bbox,
        )
        # The box is expressed in its oriented-bounding-box frame, while the
        # axis must show the canonical mesh frame returned by TF/the service.
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=pose_matrix,
            scale=0.1,
            K=K_scaled,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

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
            # prepare_frame() and the drawing helpers keep this image in RGB.
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
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
                self.get_logger().warn(
                    "FoundationPose estimator not initialized, skipping frame",
                    throttle_duration_sec=5.0,
                )
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
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Shutdown requested")
    finally:
        # if node.enable_visualization:
        #     cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node stopped")


if __name__ == '__main__':
    main()
