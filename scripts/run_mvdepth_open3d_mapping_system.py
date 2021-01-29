import cv2
import numpy as np
import open3d as o3d
import os

from argparse import ArgumentParser
from typing import List, Optional, Tuple

from smg.mapping import MVDepthOpen3DMappingSystem
from smg.mapping.remote import MappingServer, RGBDFrameMessageUtil
from smg.mvdepthnet import MonocularDepthEstimator
from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.utility import PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", "-o", type=str,
        help="an optional directory into which to save the sequence"
    )
    parser.add_argument(
        "--pool_empty_strategy", "-p", type=str, default="discard",
        choices=("discard", "grow", "replace_random", "wait"),
        help="the strategy to use when a frame message is received whilst a client handler's frame pool is empty"
    )
    parser.add_argument(
        "--save_mesh", action="store_true",
        help="whether to save the mesh into the output directory as well as the frames"
    )
    parser.add_argument(
        "--show_keyframes", action="store_true",
        help="whether to visualise the MVDepth keyframes"
    )
    args: dict = vars(parser.parse_args())

    output_dir: Optional[str] = args["output_dir"]
    save_mesh: bool = args["save_mesh"]
    show_keyframes: bool = args["show_keyframes"]

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.EPoolEmptyStrategy.make(args["pool_empty_strategy"])
    ) as server:
        # Construct the depth estimator.
        depth_estimator: MonocularDepthEstimator = MonocularDepthEstimator(
            "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar", debug=True
        )

        # Construct the mapping system.
        mapping_system: MVDepthOpen3DMappingSystem = MVDepthOpen3DMappingSystem(
            depth_estimator=depth_estimator, output_dir=output_dir, server=server
        )

        # Start the server.
        server.start()

        # Run the mapping system.
        tsdf: o3d.pipelines.integration.ScalableTSDFVolume = mapping_system.run()

        # Close any OpenCV windows.
        cv2.destroyAllWindows()

        # Convert the TSDF to a mesh, and visualise it alongside a voxel grid for evaluation purposes.
        mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(tsdf, print_progress=True)
        grid: o3d.geometry.LineSet = VisualisationUtil.make_voxel_grid(
            [-2, -2, -2], [2, 0, 2], [1, 1, 1]
        )
        to_visualise: List[o3d.geometry.Geometry] = [mesh, grid]

        # If requested, also show the MVDepth keyframes.
        if show_keyframes:
            keyframes: List[Tuple[np.ndarray, np.ndarray]] = depth_estimator.get_keyframes()
            to_visualise += [
                VisualisationUtil.make_axes(pose, size=0.01) for _, pose in keyframes
            ]

        # Run the Open3D visualiser.
        VisualisationUtil.visualise_geometries(to_visualise)

        # If requested, save the mesh.
        if output_dir is not None and save_mesh:
            # noinspection PyTypeChecker
            o3d.io.write_triangle_mesh(os.path.join(output_dir, "mesh.ply"), mesh, print_progress=True)


if __name__ == "__main__":
    main()
