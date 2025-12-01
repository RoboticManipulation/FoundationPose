import os
import pyrealsense2 as rs
import numpy as np
import cv2
import trimesh
from estimater import *
from datareader import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_file', type=str, default='demo_data/mustard0/mesh/textured_simple.obj', help="Path to mesh file")
    parser.add_argument('--debug', type=int, default=1, help="Debug level")
    parser.add_argument('--debug_dir', type=str, default='debug_realsense', help="Debug directory")
    args = parser.parse_args()

    # Setup debug directory
    debug_dir = args.debug_dir
    os.makedirs(debug_dir, exist_ok=True)
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    print("Starting RealSense camera...")

    profile = None

    # Try a couple of common RealSense RGBD profiles, then fall back
    for w, h in [ (848, 480)]: # (848, 480), (640, 480)
        try:
            config = rs.config()
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
            profile = pipeline.start(config)
            print(f"RealSense started with {w}x{h} depth/color @ 30 FPS")
            break
        except RuntimeError as e:
            print(f"Failed to start RealSense with {w}x{h} streams: {e}")
        except Exception as e:
            print(f"Configuration {w}x{h} not supported: {e}")

    if profile is None:
        # Last resort: let RealSense choose a default configuration
        try:
            profile = pipeline.start()
            print("RealSense started with default configuration")
        except RuntimeError as e:
            raise RuntimeError(
                f"Unable to start RealSense pipeline with any configuration: {e}"
            )

    # Get camera intrinsics (requires a color stream to be available)
    try:
        color_profile = profile.get_stream(rs.stream.color)
    except RuntimeError as e:
        pipeline.stop()
        raise RuntimeError(
            "RealSense pipeline started but no color stream is available. "
            "This demo expects an RGB-D camera; if you are using a depth-only "
            f"device (e.g. certain D40x modules), it cannot run as written. "
            f"Details: {e}"
        )

    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    
    K = np.array([
        [color_intrinsics.fx, 0, color_intrinsics.ppx],
        [0, color_intrinsics.fy, color_intrinsics.ppy],
        [0, 0, 1]
    ])
    
    print(f"Camera intrinsics:\n{K}")
    
    # Align depth to color
    align = rs.align(rs.stream.color)
    
    # Load mesh
    mesh = trimesh.load(args.mesh_file)

    # Compute oriented bounding box
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # Setup FoundationPose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=args.debug,
        glctx=glctx
    )
    
    print("FoundationPose initialized. Press 'q' to quit, 's' to save current frame, 'r' to reset tracking")
    
    pose = None
    frame_count = 0
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert to numpy arrays
            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            
            # Convert depth from mm to meters
            depth = depth.astype(np.float32) / 1000.0
            
            # Create mask (for first frame, use full image or segment manually)
            if pose is None:
                # Initial pose estimation - you need a mask here
                # For now, using a simple center region as mask
                mask = np.zeros(color.shape[:2], dtype=np.uint8)
                h, w = mask.shape
                mask[h//4:3*h//4, w//4:3*w//4] = 255
                
                print(f"Running initial pose estimation...")
                pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
                
                if pose is None:
                    print("Failed to estimate initial pose, skipping frame")
                    continue
            else:
                # Track with refiner
                pose = est.track_one(rgb=color, depth=depth, K=K, iteration=2)
            
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
            elif key == ord('s'):
                save_path = f"{debug_dir}/frame_{frame_count:04d}.png"
                cv2.imwrite(save_path, vis)
                print(f"Saved frame to {save_path}")
            
            frame_count += 1
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")
