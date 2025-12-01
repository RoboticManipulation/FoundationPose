import numpy as np
import cv2
import trimesh
import argparse
import threading
import torch

# Import FoundationPose modules FIRST 
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
import pyrealsense2 as rs

# TF2 imports for publishing pose as transform
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

class FoundationPoseROS2(Node):
    def __init__(self, args):
        super().__init__('foundationpose_ros2')
        
        self.args = args
        self.bridge = CvBridge()
        
        # Setup debug directory
        self.debug_dir = args.debug_dir
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
        
        # Threading lock for pose updates
        self.lock = threading.Lock()
        
        # TF2 broadcaster for publishing pose
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.camera_frame = args.camera_frame  # Parent frame (camera)
        self.object_frame = args.object_frame  # Child frame (detected object)
        
        # Load mesh
        self.get_logger().info(f"Loading mesh from {args.mesh_file}")
        self.mesh = trimesh.load(args.mesh_file)
        
        # Compute oriented bounding box
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        
        # Setup FoundationPose
        self.get_logger().info("Initializing FoundationPose...")
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
            debug=args.debug,
            glctx=glctx
        )
        
        # QoS profile to match publisher (RELIABLE, not BEST_EFFORT)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to camera info to get intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Synchronized subscribers for color and depth images (RealSense)
        self.color_sub = message_filters.Subscriber(
            self,
            ROSImage,
            '/camera/camera/color/image_rect_raw',
            qos_profile=qos
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            ROSImage,
            '/camera/camera/aligned_depth_to_color/image_raw',
            qos_profile=qos
        )
        
        # self.camera_info_sub = self.create_subscription(
        #     CameraInfo,
        #     '/sim_camera_info',
        #     self.camera_info_callback,
        #     10
        # )
        
        # # Synchronized subscribers for color and depth images
        # self.color_sub = message_filters.Subscriber(
        #     self,
        #     ROSImage,
        #     '/sim_camera_rgb',
        #     qos_profile=qos
        # )
        # self.depth_sub = message_filters.Subscriber(
        #     self,
        #     ROSImage,
        #     '/sim_camera_depth',
        #     qos_profile=qos
        # )
        
        # Time synchronizer for color and depth
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=20,
            slop=0.3
        )
        self.sync.registerCallback(self.image_callback)
        
        self.get_logger().info("FoundationPose ROS2 node initialized")
        self.get_logger().info(f"Publishing TF: {self.camera_frame} -> {self.object_frame}")
        self.get_logger().info("Waiting for camera intrinsics...")
        self.get_logger().info("Press 'q' to quit, 's' to save current frame, 'r' to reset tracking")
    
    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo message"""
        if not self.intrinsics_received:
            # Log full CameraInfo for debugging
            self.get_logger().info(f"Image resolution from CameraInfo: {msg.width}x{msg.height}")

            # Extract K matrix (intrinsic camera matrix)
            k_mat = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"K matrix (intrinsic):\n{k_mat}")

            # # Extract P matrix (projection matrix for rectified images)
            # p_mat = np.array(msg.p, dtype=np.float64).reshape(3, 4)
            # self.get_logger().info(f"P matrix (projection):\n{p_mat}")
            
            # Extract Distortion coefficients
            # self.D = np.array(msg.d, dtype=np.float64)
            # self.get_logger().info(f"Distortion coefficients:\n{self.D}")

            # For aligned depth images, use the K matrix directly
            # The K matrix represents the actual camera intrinsics
            self.K = k_mat

            self.intrinsics_received = True
            self.get_logger().info(f"Using K matrix as camera intrinsics:\n{self.K}")
            self.get_logger().info(f"K dtype: {self.K.dtype}")
    
    def publish_pose_tf(self, pose_matrix, timestamp):
        """Publish the pose as a TF2 transform
        
        Args:
            pose_matrix: 4x4 transformation matrix (object in camera frame)
            timestamp: ROS2 timestamp from the image message
        """
        # Create transform message
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.object_frame
        # t.header.frame_id =  self.object_frame
        # t.child_frame_id = self.camera_frame
        
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
        if not self.intrinsics_received:
            self.get_logger().warn("Camera intrinsics not yet received, skipping frame")
            return
        
        try:
            # Convert ROS Image messages to OpenCV/numpy format
            # Color image - RealSense ROS2 typically outputs rgb8, convert to bgr8 to match pyrealsense2
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
            
            # Ensure color is contiguous and proper format (matches pyrealsense2 np.asanyarray behavior)
            color = np.ascontiguousarray(color)
            
            # Depth image handling - format depends on source
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Ensure depth is 2D (squeeze extra dimensions if present)
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]  # Take first channel if 3D
            depth = np.squeeze(depth)  # Remove any singleton dimensions
            
            # Ensure depth is contiguous (matches pyrealsense2 np.asanyarray behavior)
            depth = np.ascontiguousarray(depth)
            
            # Handle depth conversion based on encoding:
            # - 16UC1: 16-bit unsigned int in millimeters (RealSense) → divide by 1000
            # - 32FC1: 32-bit float already in meters (Isaac Sim) → no conversion needed
            if depth_msg.encoding == '16UC1':
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)
            
            # Handle invalid depth values (Isaac Sim uses inf for no depth)
            depth = np.where(np.isinf(depth) | np.isnan(depth), 0.0, depth)
            
            # Downscale images if requested (saves GPU memory)
            scale = self.args.downscale
            K_scaled = self.K.copy()
            if scale != 1.0:
                new_h, new_w = int(color.shape[0] * scale), int(color.shape[1] * scale)
                color = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                # Scale intrinsics (fx, fy, cx, cy)
                K_scaled[0, 0] *= scale  # fx
                K_scaled[1, 1] *= scale  # fy
                K_scaled[0, 2] *= scale  # cx
                K_scaled[1, 2] *= scale  # cy
            
            # UNDISTORT DEPTH to align with Rectified Color
            # Using K as NewCameraMatrix because P(3x3) == K
            # if self.D is not None and self.K is not None:
            #      depth = cv2.undistort(depth, self.K, self.D, None, self.K)

            # Debug: Log once to verify formats
            if self.frame_count == 0:
                self.get_logger().info(f"Color shape: {color.shape}, dtype: {color.dtype}, encoding: {color_msg.encoding} ")
                self.get_logger().info(f"Depth shape: {depth.shape}, dtype: {depth.dtype}, encoding: {depth_msg.encoding} ")
                self.get_logger().info(f"Depth range: min={depth.min():.3f}m, max={depth.max():.3f}m")
                self.get_logger().info(f"Color image size: {color_msg.width}x{color_msg.height}")
                self.get_logger().info(f"Depth image size: {depth_msg.width}x{depth_msg.height}")
                if scale != 1.0:
                    self.get_logger().info(f"Downscaled to: {color.shape[1]}x{color.shape[0]} (scale={scale})")
            
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return
        
        with self.lock:
            # Pose estimation
            if self.pose is None:
                # Initial pose estimation - use larger mask (80% of image) to increase chance of valid depth
                mask = np.zeros(color.shape[:2], dtype=np.uint8)
                h, w = mask.shape
                margin_h, margin_w = h // 10, w // 10  # 10% margin on each side
                mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
                
                # Check how much valid depth we have in the masked region
                valid_depth_in_mask = np.sum((depth > 0.1) & (depth < 3.0) & (mask > 0))
                self.get_logger().info(f"Valid depth points in mask: {valid_depth_in_mask}")
                
                if valid_depth_in_mask < 1000:
                    self.get_logger().warn(f"Not enough valid depth points ({valid_depth_in_mask}), skipping frame")
                    return
                
                self.get_logger().info("Running initial pose estimation...")
                try:
                    # Clear CUDA cache to free memory before heavy computation
                    torch.cuda.empty_cache()
                    self.pose = self.est.register(K=K_scaled, rgb=color, depth=depth, ob_mask=mask, iteration=3)
                except Exception as e:
                    self.get_logger().error(f"Registration failed with exception: {e}")
                    torch.cuda.empty_cache()  # Try to recover memory
                    self.pose = None
                
                if self.pose is None:
                    self.get_logger().warn("Failed to estimate initial pose, skipping frame")
                    return
                    
                self.get_logger().info("Initial pose registration successful!")
            else:
                # Track with refiner
                try:
                    self.pose = self.est.track_one(rgb=color, depth=depth, K=K_scaled, iteration=2)
                except RuntimeError as e:
                    self.get_logger().warn(f"Tracking failed: {e}, resetting pose...")
                    self.pose = None
                    return
            
            # Visualize
            center_pose = self.pose @ np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(K_scaled, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K_scaled, thickness=3, transparency=0, is_input_rgb=True)
            
            # Publish pose as TF2 transform
            self.publish_pose_tf(center_pose, color_msg.header.stamp)
            
            cv2.imshow('FoundationPose Tracking (ROS2)', vis)
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_file', type=str, default='demo_data/mustard0/mesh/textured_simple.obj', help="Path to mesh file")
    parser.add_argument('--debug', type=int, default=1, help="Debug level")
    parser.add_argument('--debug_dir', type=str, default='debug_ros2', help="Debug directory")
    parser.add_argument('--downscale', type=float, default=1.0, help="Downscale factor for images (e.g., 0.5 = half resolution, saves GPU memory)")
    parser.add_argument('--camera_frame', type=str, default='camera_color_optical_frame', help="TF2 parent frame (camera frame)")
    parser.add_argument('--object_frame', type=str, default='foundationpose_object', help="TF2 child frame (object frame)")
    
    # Parse known args to allow ROS2 args to pass through
    parsed_args, _ = parser.parse_known_args()
    
    rclpy.init(args=args)
    
    node = FoundationPoseROS2(parsed_args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node stopped")


if __name__ == '__main__':
    main()
