import numpy as np
import cv2
import trimesh
import argparse
import threading

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
        self.intrinsics_received = False
        
        # FoundationPose state
        self.pose = None
        self.frame_count = 0
        self.est = None
        self.to_origin = None
        self.bbox = None
        
        # Threading lock for pose updates
        self.lock = threading.Lock()
        
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
        
        # Subscribe to camera info to get intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Synchronized subscribers for color and depth images
        self.color_sub = message_filters.Subscriber(
            self,
            ROSImage,
            '/camera/camera/color/image_rect_raw',
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            ROSImage,
            '/camera/camera/aligned_depth_to_color/image_raw',
        )
        
        # Time synchronizer for color and depth
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)
        
        self.get_logger().info("FoundationPose ROS2 node initialized")
        self.get_logger().info("Waiting for camera intrinsics...")
        self.get_logger().info("Press 'q' to quit, 's' to save current frame, 'r' to reset tracking")
    
    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo message"""
        if not self.intrinsics_received:
            # When using rectified images (image_rect_raw), we MUST use the Projection matrix (P)
            # instead of the raw intrinsics (K). P is 3x4, we want the 3x3 intrinsic part.
            # P = [fx' 0  cx' Tx]
            #     [0  fy' cy' Ty]
            # #     [0   0   1   0]
            p_mat = np.array(msg.p, dtype=np.float64).reshape(3, 4)
            self.K = p_mat[:3, :3]
            
            self.intrinsics_received = True
            self.get_logger().info(f"Camera intrinsics received (from Projection Matrix P):\n{self.K}")
            self.get_logger().info(f"K dtype: {self.K.dtype}")
    
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
            
            # Depth image - typically 16UC1 (16-bit unsigned, in mm)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Ensure depth is 2D (squeeze extra dimensions if present)
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]  # Take first channel if 3D
            depth = np.squeeze(depth)  # Remove any singleton dimensions
            
            # Ensure depth is contiguous (matches pyrealsense2 np.asanyarray behavior)
            depth = np.ascontiguousarray(depth)
            
            # Convert depth from mm to meters (same as pyrealsense2 format)
            depth = depth.astype(np.float32) / 1000.0
            
            # Debug: Log once to verify formats
            if self.frame_count == 0:
                self.get_logger().info(f"Color shape: {color.shape}, dtype: {color.dtype}, encoding: {color_msg.encoding}")
                self.get_logger().info(f"Depth shape: {depth.shape}, dtype: {depth.dtype}, encoding: {depth_msg.encoding}")
                self.get_logger().info(f"Depth range: min={depth.min():.3f}m, max={depth.max():.3f}m")
            
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return
        
        with self.lock:
            # Pose estimation
            if self.pose is None:
                # Initial pose estimation - using center region as mask
                mask = np.zeros(color.shape[:2], dtype=np.uint8)
                h, w = mask.shape
                mask[h//4:3*h//4, w//4:3*w//4] = 255
                
                self.get_logger().info("Running initial pose estimation...")
                self.pose = self.est.register(K=self.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
                
                if self.pose is None:
                    self.get_logger().warn("Failed to estimate initial pose, skipping frame")
                    return
            else:
                # Track with refiner
                self.pose = self.est.track_one(rgb=color, depth=depth, K=self.K, iteration=2)
            
            # Visualize
            center_pose = self.pose @ np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
            
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
