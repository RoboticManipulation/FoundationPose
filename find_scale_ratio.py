#!/usr/bin/env python3
"""
# reference mesh that has correct real-world dimensions.

conda run -n foundationpose_ros python find_scale_ratio.py \
    --reference textured_simple.obj \
    --target ob_0000005.obj

# use the output with scale_mesh.py:
conda run -n foundationpose_ros python scale_mesh.py \
    --input ob_0000005.obj \
    --output ob_0000005_scaled.obj \
    --scale <ratio_from_this_script>
"""
import trimesh
import argparse

def find_scale_ratio(reference_file, target_file, axis='max'):
    """
    Find scale ratio to make target mesh match reference mesh size.

    Args:
        reference_file: Mesh with correct real-world scale
        target_file: Mesh to be scaled (e.g., from Sam3D Objects)
        axis: Which dimension to compare ('max', 'width', 'height', 'depth')

    Returns:
        scale_factor: Multiply target by this to match reference
    """
    print("Loading meshes...")
    ref_mesh = trimesh.load(reference_file)
    target_mesh = trimesh.load(target_file)

    print("\n" + "=" * 60)
    print("REFERENCE MESH (correct scale)")
    print("=" * 60)
    print(f"File: {reference_file}")
    print(f"Vertices: {len(ref_mesh.vertices)}")
    print(f"Extents:")
    print(f"  Width (X):  {ref_mesh.extents[0]:.6f} units ({ref_mesh.extents[0]*100:.2f} cm if in meters)")
    print(f"  Height (Y): {ref_mesh.extents[1]:.6f} units ({ref_mesh.extents[1]*100:.2f} cm if in meters)")
    print(f"  Depth (Z):  {ref_mesh.extents[2]:.6f} units ({ref_mesh.extents[2]*100:.2f} cm if in meters)")
    print(f"  Max:        {ref_mesh.extents.max():.6f} units")

    print("\n" + "=" * 60)
    print("TARGET MESH (to be scaled)")
    print("=" * 60)
    print(f"File: {target_file}")
    print(f"Vertices: {len(target_mesh.vertices)}")
    print(f"Extents:")
    print(f"  Width (X):  {target_mesh.extents[0]:.6f} units")
    print(f"  Height (Y): {target_mesh.extents[1]:.6f} units")
    print(f"  Depth (Z):  {target_mesh.extents[2]:.6f} units")
    print(f"  Max:        {target_mesh.extents.max():.6f} units")

    # Calculate scale ratio based on chosen axis
    axis_map = {
        'width': 0,
        'height': 1,
        'depth': 2,
        'max': -1
    }

    if axis not in axis_map:
        raise ValueError(f"axis must be one of {list(axis_map.keys())}")

    if axis == 'max':
        ref_size = ref_mesh.extents.max()
        target_size = target_mesh.extents.max()
    else:
        axis_idx = axis_map[axis]
        ref_size = ref_mesh.extents[axis_idx]
        target_size = target_mesh.extents[axis_idx]

    scale_ratio = ref_size / target_size

    print("\n" + "=" * 60)
    print("SCALE RATIO CALCULATION")
    print("=" * 60)
    print(f"Comparing: {axis}")
    print(f"  Reference size: {ref_size:.6f}")
    print(f"  Target size:    {target_size:.6f}")
    print(f"  Ratio:          {scale_ratio:.6f}")
    print(f"\nTarget is {target_size/ref_size:.2f}x {'larger' if target_size > ref_size else 'smaller'} than reference")

    print("\n" + "=" * 60)
    print("HOW TO USE THIS RATIO")
    print("=" * 60)
    print(f"Run this command to scale your target mesh:\n")
    print(f"  python scale_mesh.py \\")
    print(f"      --input {target_file} \\")
    print(f"      --output {target_file.replace('.obj', '_scaled.obj')} \\")
    print(f"      --scale {scale_ratio:.6f}")
    print()

    # Show what the scaled mesh would look like
    print("=" * 60)
    print("AFTER SCALING (preview)")
    print("=" * 60)
    scaled_extents = target_mesh.extents * scale_ratio
    print(f"Target mesh will have extents:")
    print(f"  Width (X):  {scaled_extents[0]:.6f} (reference: {ref_mesh.extents[0]:.6f})")
    print(f"  Height (Y): {scaled_extents[1]:.6f} (reference: {ref_mesh.extents[1]:.6f})")
    print(f"  Depth (Z):  {scaled_extents[2]:.6f} (reference: {ref_mesh.extents[2]:.6f})")
    print(f"  Max:        {scaled_extents.max():.6f} (reference: {ref_mesh.extents.max():.6f})")

    return scale_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find scale ratio between two meshes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

1. Find the scale ratio:
   python find_scale_ratio.py --reference textured_simple.obj --target ob_0000005.obj

2. Use the ratio to scale the target mesh:
   python scale_mesh.py --input ob_0000005.obj --output ob_0000005_scaled.obj --scale 0.191359

        """
    )
    parser.add_argument('--reference', required=True,
                       help='Reference mesh with correct scale (e.g., textured_simple.obj)')
    parser.add_argument('--target', required=True,
                       help='Target mesh to be scaled (e.g., from Sam3D Objects)')
    parser.add_argument('--axis', default='max',
                       choices=['width', 'height', 'depth', 'max'],
                       help='Which dimension to compare (default: max)')

    args = parser.parse_args()

    ratio = find_scale_ratio(args.reference, args.target, args.axis)

    print(f"\n✓ Scale ratio: {ratio:.6f}")
