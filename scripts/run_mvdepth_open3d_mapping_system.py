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
        "--detect_objects", "-d", action="store_true",
        help="whether to detect 3D objects"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        help="an optional directory into which to save the sequence"
    )
    parser.add_argument(
        "--pool_empty_strategy", "-p", type=str, default="replace_random",
        choices=("discard", "grow", "replace_random", "wait"),
        help="the strategy to use when a frame message is received whilst a client handler's frame pool is empty"
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="whether to save the sequence of frames used to reconstruct the TSDF"
    )
    parser.add_argument(
        "--save_reconstruction", action="store_true",
        help="whether to save the reconstructed mesh"
    )
    parser.add_argument(
        "--show_keyframes", action="store_true",
        help="whether to visualise the MVDepth keyframes"
    )
    args: dict = vars(parser.parse_args())

    output_dir: Optional[str] = args["output_dir"]

    # Construct the depth estimator.
    depth_estimator: MonocularDepthEstimator = MonocularDepthEstimator(
        "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar", debug=True
    )

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.EPoolEmptyStrategy.make(args["pool_empty_strategy"])
    ) as server:
        # Construct the mapping system.
        mapping_system: MVDepthOpen3DMappingSystem = MVDepthOpen3DMappingSystem(
            server, depth_estimator, detect_objects=args["detect_objects"], output_dir=output_dir,
            save_frames=args["save_frames"]
        )

        # Start the server.
        server.start()

        # Run the mapping system.
        tsdf, objects = mapping_system.run()

        # Destroy any remaining OpenCV windows.
        cv2.destroyAllWindows()

        # Visualise the reconstructed map.
        grid: o3d.geometry.LineSet = VisualisationUtil.make_voxel_grid([-3, -2, -3], [3, 0, 3], [1, 1, 1])
        mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(tsdf, print_progress=True)
        to_visualise: List[o3d.geometry.Geometry] = [grid, mesh]

        for obj in objects:
            box: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.AxisAlignedBoundingBox(*obj.box_3d)
            box.color = (1.0, 0.0, 1.0)
            to_visualise.append(box)

        if args["show_keyframes"]:
            keyframes: List[Tuple[np.ndarray, np.ndarray]] = depth_estimator.get_keyframes()
            to_visualise += [
                VisualisationUtil.make_axes(pose, size=0.01) for _, pose in keyframes
            ]

        VisualisationUtil.visualise_geometries(to_visualise)

        # If an output directory has been specified and we're saving the reconstruction, save it now.
        if output_dir is not None and args["save_reconstruction"]:
            # noinspection PyTypeChecker
            o3d.io.write_triangle_mesh(os.path.join(output_dir, "mesh.ply"), mesh, print_progress=True)


if __name__ == "__main__":
    main()
