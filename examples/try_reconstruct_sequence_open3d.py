import cv2
import numpy as np
import open3d as o3d
import os

from argparse import ArgumentParser
from typing import Tuple

from smg.open3d import ReconstructionUtil, VisualisationUtil
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
        o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 532.5694641250893, 531.5410880910171, 320.0, 240.0
        )
    elif source_type == "tello":
        sequence_dir: str = "C:/smglib/smg-mapping/output-tello"
        intrinsics: Tuple[float, float, float, float] = (946.60441222, 941.38386885, 460.29254907, 357.08431882)
        o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
            960, 720, 946.60441222, 941.38386885, 460.29254907, 357.08431882
        )
    else:
        raise RuntimeError(f"Unknown source type: {source_type}")

    # Make the initial TSDF.
    tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.005,
        sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

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

        colour_image: np.ndarray = cv2.imread(colour_filename)
        convergence_map: np.ndarray = cv2.imread(convergence_filename, cv2.IMREAD_UNCHANGED)
        depth_image: np.ndarray = ImageUtil.load_depth_image(depth_filename)
        pose: np.ndarray = np.linalg.inv(PoseUtil.load_pose(pose_filename))

        # Post-process the depth image to reduce noise.
        depth_image = DepthProcessor.postprocess_depth(depth_image, convergence_map, intrinsics)

        # Visualise the keyframe as a coloured 3D point cloud.
        # VisualisationUtil.visualise_rgbd_image(colour_image, depth_image, intrinsics)

        # Fuse the keyframe into the TSDF.
        ReconstructionUtil.integrate_frame(
            ImageUtil.flip_channels(colour_image), depth_image, pose, o3d_intrinsics, tsdf
        )

        # Visualise the current state of the map as a mesh.
        # VisualisationUtil.visualise_geometry(ReconstructionUtil.make_mesh(tsdf))

        # Visualise the current state of the map as a voxel grid.
        # noinspection PyArgumentList
        # VisualisationUtil.visualise_geometry(o3d.geometry.VoxelGrid.create_from_point_cloud(
        #     tsdf.extract_point_cloud(), voxel_size=0.01
        # ))

        # Increment the frame index.
        frame_idx += 1

    # Visualise the final state of the map as a voxel grid.
    # noinspection PyArgumentList
    VisualisationUtil.visualise_geometry(o3d.geometry.VoxelGrid.create_from_point_cloud(
        tsdf.extract_point_cloud(), voxel_size=0.01
    ))


if __name__ == "__main__":
    main()
