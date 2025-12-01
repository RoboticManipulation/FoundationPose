import os
import numpy as np
import cv2
import trimesh
from estimater import *
from datareader import *
import argparse
import threading
import gc
import torch

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters


class FoundationPoseROS2Node(Node):
    def __init__(self, args):
        super().__init__('foundation_pose_node')
        
        self.args = args
        self.bridge = CvBridge()
        
        # Latest synchronized data
        self.latest_rgb = None
        self.latest_depth = None
        self.K = None
        self.data_lock = threading.Lock()
        self.new_data_available = False
        
        # Setup debug directory
        self.debug_dir = args.debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Subscribe to camera info first to get intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Use message_filters to synchronize RGB and depth
        self.rgb_sub = message_filters.Subscriber(self, Image, args.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, args.depth_topic)
        
        # Approximate time synchronizer (use exact if timestamps match perfectly)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.sync.registerCallback(self.synced_callback)
        
        self.get_logger().info(f"Subscribing to RGB topic: {args.rgb_topic}")
        self.get_logger().info(f"Subscribing to Depth topic: {args.depth_topic}")
        self.get_logger().info(f"Subscribing to CameraInfo topic: {args.camera_info_topic}")
        
    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo message"""
        if self.K is None:
            # K matrix is stored as a flat array [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"Received camera intrinsics:\n{self.K}")
            
    def synced_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and depth images"""
        try:
            # Convert RGB image
            if rgb_msg.encoding == 'rgb8':
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif rgb_msg.encoding == 'bgr8':
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            else:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'passthrough')
                if len(rgb.shape) == 2:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            
            # Convert depth image
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            
            # ACHTUNG : Added below to make the real ros2 work
            # Ensure depth is 2D (squeeze extra channel dimension if present)
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            
            # Handle different depth encodings
            if depth_msg.encoding == '16UC1':
                # Depth in mm, convert to meters
                depth = depth.astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                # Already in meters
                depth = depth.astype(np.float32)
            else:
                # Assume meters if unknown
                depth = depth.astype(np.float32)
            
            with self.data_lock:
                self.latest_rgb = rgb
                self.latest_depth = depth
                self.new_data_available = True
                
        except Exception as e:
            self.get_logger().error(f"Error processing images: {e}")
            
    def get_latest_data(self):
        """Get the latest synchronized RGB and depth images"""
        with self.data_lock:
            if self.new_data_available:
                self.new_data_available = False
                return self.latest_rgb.copy(), self.latest_depth.copy()
            return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_file', type=str, default='demo_data/mustard0/mesh/textured_simple.obj', help="Path to mesh file")
    parser.add_argument('--debug', type=int, default=1, help="Debug level")
    parser.add_argument('--debug_dir', type=str, default='debug_ros2', help="Debug directory")
    
    # ROS2 topic arguments - configurable
    # parser.add_argument('--rgb_topic', type=str, default='/sim_camera_rgb', help="RGB image topic")
    # parser.add_argument('--depth_topic', type=str, default='/sim_camera_depth', help="Depth image topic")
    # parser.add_argument('--camera_info_topic', type=str, default='/sim_camera_info', help="Camera info topic")
    
    parser.add_argument('--rgb_topic', type=str, default='/camera/camera/color/image_rect_raw', help="RGB image topic")
    parser.add_argument('--depth_topic', type=str, default='/camera/camera/aligned_depth_to_color/image_raw', help="Depth image topic")
    parser.add_argument('--camera_info_topic', type=str, default='/camera/camera/color/camera_info', help="Camera info topic")
    
    # Memory management arguments
    parser.add_argument('--register_iter', type=int, default=3, help="Iterations for initial registration (lower = less memory)")
    parser.add_argument('--track_iter', type=int, default=2, help="Iterations for tracking (lower = less memory)")
    parser.add_argument('--half_precision', action='store_true', help="Use half precision (float16) to reduce memory usage")
    
    # Rotation grid parameters (controls number of pose hypotheses - major memory impact!)
    # Default: min_n_views=40, inplane_step=60 -> 252 poses -> ~10GB memory
    # Low memory: min_n_views=20, inplane_step=120 -> ~60 poses -> ~3GB memory
    parser.add_argument('--min_n_views', type=int, default=20, help="Min views for rotation grid (default 20, original 40)")
    parser.add_argument('--inplane_step', type=int, default=120, help="In-plane rotation step in degrees (default 120, original 60)")
    
    args = parser.parse_args()
    
    # Set memory allocation config to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Clear CUDA cache at startup
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    # Initialize ROS2
    rclpy.init()
    node = FoundationPoseROS2Node(args)
    
    # Spin ROS2 in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    # Wait for camera intrinsics
    print("Waiting for camera intrinsics...")
    while node.K is None and rclpy.ok():
        import time
        time.sleep(0.1)
    
    if node.K is None:
        print("Failed to get camera intrinsics, exiting...")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    K = node.K
    print(f"Camera intrinsics:\n{K}")
    
    # Load mesh
    mesh = trimesh.load(args.mesh_file)

    # Compute oriented bounding box
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    # Setup FoundationPose
    print("Loading scorer model...")
    scorer = ScorePredictor()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"After scorer - GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    
    print("Loading refiner model...")
    refiner = PoseRefinePredictor()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"After refiner - GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    
    # Convert models to half precision if requested (saves ~50% memory)
    if args.half_precision:
        print("Converting models to half precision (float16)...")
        scorer.model = scorer.model.half()
        refiner.model = refiner.model.half()
        torch.cuda.empty_cache()
        gc.collect()
        print(f"After half precision - GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx
    )
    
    # Regenerate rotation grid with reduced pose hypotheses for lower memory usage
    # Original: min_n_views=40, inplane_step=60 -> 252 poses
    # Reduced: min_n_views=20, inplane_step=120 -> ~60 poses
    print(f"Regenerating rotation grid with min_n_views={args.min_n_views}, inplane_step={args.inplane_step}...")
    est.make_rotation_grid(min_n_views=args.min_n_views, inplane_step=args.inplane_step)
    print(f"Rotation grid size: {est.rot_grid.shape[0]} pose hypotheses")
    
    # Clear CUDA cache after model initialization
    torch.cuda.empty_cache()
    gc.collect()
    print(f"After init - GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    
    print("FoundationPose initialized. Press 'q' to quit, 's' to save current frame, 'r' to reset tracking")
    print("Waiting for images from ROS2 topics...")
    
    pose = None
    frame_count = 0
    
    try:
        while rclpy.ok():
            # Get latest synchronized data
            color, depth = node.get_latest_data()
            
            if color is None or depth is None:
                cv2.waitKey(10)
                continue
            
            # Create mask (for first frame, use full image or segment manually)
            if pose is None:
                # Initial pose estimation - you need a mask here
                # For now, using a simple center region as mask
                mask = np.zeros(color.shape[:2], dtype=np.uint8)
                h, w = mask.shape
                mask[h//4:3*h//4, w//4:3*w//4] = 255
                # mask = np.ones(color.shape[:2], dtype=np.uint8) * 255
                
                print(f"Running initial pose estimation...")
                # Clear cache before heavy operation
                torch.cuda.empty_cache()
                gc.collect()
                
                pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.register_iter)
                
                # Clear cache after registration
                torch.cuda.empty_cache()
                gc.collect()
                
                if pose is None:
                    print("Failed to estimate initial pose, skipping frame")
                    continue
            else:
                # Track with refiner
                pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_iter)
            
            # Periodic memory cleanup (every 100 frames)
            if frame_count % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Visualize
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            
            cv2.imshow('FoundationPose Tracking', vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting tracking...")
                pose = None
                torch.cuda.empty_cache()
                gc.collect()
            elif key == ord('s'):
                save_path = f"{args.debug_dir}/frame_{frame_count:04d}.png"
                cv2.imwrite(save_path, vis)
                print(f"Saved frame to {save_path}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
        print("ROS2 node stopped")


if __name__ == '__main__':
    main()
