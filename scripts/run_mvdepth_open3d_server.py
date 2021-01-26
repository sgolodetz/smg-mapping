import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.mapping.remote import MappingServer, RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.mvdepthnet import MonocularDepthEstimator
from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.opengl import OpenGLImageRenderer
from smg.utility import GeometryUtil, ImageUtil, PooledQueue, RGBDSequenceUtil


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

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("MVDepth -> Open3D Server")

    # Create the TSDF.
    tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.EPoolEmptyStrategy.make(args["pool_empty_strategy"])
    ) as server:
        # Construct the image renderer.
        with OpenGLImageRenderer() as image_renderer:
            client_id: int = 0
            colour_image: Optional[np.ndarray] = None
            depth_estimator: MonocularDepthEstimator = MonocularDepthEstimator(
                "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar", debug=True
            )
            frame_idx: int = 0
            receiver: RGBDFrameReceiver = RGBDFrameReceiver()

            # Start the server.
            server.start()

            while True:
                # Process any PyGame events.
                for event in pygame.event.get():
                    # If the user wants to quit:
                    if event.type == pygame.QUIT:
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

                        VisualisationUtil.visualise_geometries(to_visualise)

                        # If requested, save the mesh.
                        if output_dir is not None and save_mesh:
                            # noinspection PyTypeChecker
                            o3d.io.write_triangle_mesh(os.path.join(output_dir, "mesh.ply"), mesh, print_progress=True)

                        # Shut down pygame, and forcibly exit the program.
                        pygame.quit()
                        # noinspection PyProtectedMember
                        os._exit(0)

                # If the server has a frame from the client that has not yet been processed:
                if server.has_frames_now(client_id):
                    # Get the camera intrinsics from the server, and pass them to the depth estimator.
                    intrinsics: Tuple[float, float, float, float] = server.get_intrinsics(client_id)[0]
                    depth_estimator.set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                    # Get the frame from the server.
                    server.get_frame(client_id, receiver)
                    colour_image = receiver.get_rgb_image()
                    tracker_w_t_c: np.ndarray = receiver.get_pose()

                    # If an output directory has been specified, save the frame to disk.
                    if output_dir is not None:
                        depth_image: np.ndarray = receiver.get_depth_image()
                        RGBDSequenceUtil.save_frame(
                            frame_idx, output_dir, colour_image, depth_image, tracker_w_t_c,
                            colour_intrinsics=intrinsics, depth_intrinsics=intrinsics
                        )
                        frame_idx += 1

                    # Try to estimate a depth image for the frame.
                    estimated_depth_image: Optional[np.ndarray] = depth_estimator.estimate_depth(
                        colour_image, tracker_w_t_c
                    )

                    # If a depth image was successfully estimated:
                    if estimated_depth_image is not None:
                        # Limit its range to 3m (more distant points can be unreliable).
                        estimated_depth_image = np.where(estimated_depth_image <= 3.0, estimated_depth_image, 0.0)

                        # Fuse the frame into the TSDF.
                        start = timer()

                        height, width = estimated_depth_image.shape
                        fx, fy, cx, cy = intrinsics
                        o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                            width, height, fx, fy, cx, cy
                        )
                        ReconstructionUtil.integrate_frame(
                            ImageUtil.flip_channels(colour_image), estimated_depth_image, np.linalg.inv(tracker_w_t_c),
                            o3d_intrinsics, tsdf
                        )

                        end = timer()
                        print(f"  - Time: {end - start}s")

                # Clear the colour buffer.
                glClearColor(1.0, 1.0, 1.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)

                # If a colour image is available, draw it.
                if colour_image is not None:
                    image_renderer.render_image(ImageUtil.flip_channels(colour_image))

                # Swap the front and back buffers.
                pygame.display.flip()


if __name__ == "__main__":
    main()
