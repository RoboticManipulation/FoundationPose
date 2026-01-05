#!/usr/bin/env python3
"""
Scale a mesh to match target size for FoundationPose
"""
import trimesh
import argparse

def scale_mesh(input_file, output_file, scale_factor):
    """Scale mesh by given factor"""
    print(f"Loading mesh from: {input_file}")
    mesh = trimesh.load(input_file)

    print(f"Original extents: {mesh.extents}")
    print(f"Original max extent: {mesh.extents.max():.6f}")

    # Scale vertices
    mesh.vertices *= scale_factor

    print(f"Scaled extents: {mesh.extents}")
    print(f"Scaled max extent: {mesh.extents.max():.6f}")

    print(f"Saving to: {output_file}")
    mesh.export(output_file)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input mesh file')
    parser.add_argument('--output', type=str, required=True, help='Output mesh file')
    parser.add_argument('--scale', type=float, required=True, help='Scale factor (e.g., 0.1915 to make 5x smaller)')

    args = parser.parse_args()
    scale_mesh(args.input, args.output, args.scale)
