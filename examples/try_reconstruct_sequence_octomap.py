import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Tuple

from smg.pyoctomap import *
from smg.pyremode import DepthProcessor
from smg.utility import ImageUtil, PoseUtil


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--source_type", "-t", type=str, required=True, choices=("ardrone2", "kinect", "tello"),
        help="the source type"
    )
    args: dict = vars(parser.parse_args())

    # Set the appropriate settings for the source type.
    # FIXME: These should ultimately be loaded in rather than hard-coded.
    source_type: str = args.get("source_type")
    if source_type == "kinect":
        sequence_dir: str = "C:/smglib/smg-mapping/output-kinect"
        intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)
    elif source_type == "tello":
        sequence_dir: str = "C:/smglib/smg-mapping/output-tello"
        intrinsics: Tuple[float, float, float, float] = (946.60441222, 941.38386885, 460.29254907, 357.08431882)
    else:
        raise RuntimeError(f"Unknown source type: {source_type}")

    # Make the initial octree.
    tree: OcTree = OcTree(0.025)

    frame_idx: int = 0

    # Until we run out of keyframes:
    while True:
        # Try to load the next keyframe from disk.
        colour_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.color.png")
        convergence_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.convergence.png")
        depth_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.depth.png")
        pose_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.pose.txt")

        # If the colour image doesn't exist, early out.
        if not os.path.exists(colour_filename):
            break

        # If the colour image exists but the depth image doesn't, skip the keyframe (it's likely that
        # the user renamed the depth image to prevent this keyframe being used).
        if not os.path.exists(depth_filename):
            frame_idx += 1
            continue

        print(f"Processing frame {frame_idx}...")

        convergence_map: np.ndarray = cv2.imread(convergence_filename, cv2.IMREAD_UNCHANGED)
        depth_image: np.ndarray = ImageUtil.load_depth_image(depth_filename)
        pose: np.ndarray = PoseUtil.load_pose(pose_filename)

        # Post-process the depth image to reduce noise.
        depth_image = DepthProcessor.postprocess_depth(depth_image, convergence_map, intrinsics)

        # Make an Octomap point cloud from the depth image, and fuse it into the octree.
        pcd: Pointcloud = OctomapUtil.make_point_cloud(depth_image, pose, intrinsics)
        origin: Vector3 = Vector3(0.0, 0.0, 0.0)
        tree.insert_point_cloud(pcd, origin)

        # Increment the frame index.
        frame_idx += 1

    # Save the finished octree to disk.
    tree.write_binary("kinect_025.bt")


if __name__ == "__main__":
    main()
