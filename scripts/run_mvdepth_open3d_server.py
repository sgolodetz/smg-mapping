import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Optional, Tuple

from smg.mapping.remote import MappingServer, RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.mvdepthnet import MonocularDepthEstimator
from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.opengl import OpenGLImageRenderer
from smg.utility import GeometryUtil, ImageUtil, RGBDSequenceUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str,
        help="an optional directory into which to save the sequence"
    )
    args: dict = vars(parser.parse_args())

    output_dir: Optional[str] = args["output_dir"]

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
    with MappingServer(frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message) as server:
        # Construct the image renderer.
        with OpenGLImageRenderer() as image_renderer:
            client_id: int = 0
            colour_image: Optional[np.ndarray] = None
            depth_estimator: Optional[MonocularDepthEstimator] = None
            frame_idx: int = 0
            receiver: RGBDFrameReceiver = RGBDFrameReceiver()

            # Start the server.
            server.start()

            while True:
                # Process any PyGame events.
                for event in pygame.event.get():
                    # If the user wants to quit:
                    if event.type == pygame.QUIT:
                        # Visualise the TSDF.
                        grid: o3d.geometry.LineSet = VisualisationUtil.make_voxel_grid(
                            [-2, -2, -2], [2, 0, 2], [1, 1, 1]
                        )
                        mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(tsdf, print_progress=True)
                        VisualisationUtil.visualise_geometries([grid, mesh])

                        # Shut down pygame, and forcibly exit the program.
                        pygame.quit()
                        # noinspection PyProtectedMember
                        os._exit(0)

                # If the server has a frame from the client that has not yet been processed:
                if server.has_frames_now(client_id):
                    # Get the camera intrinsics from the server.
                    intrinsics: Tuple[float, float, float, float] = server.get_intrinsics(client_id)[0]

                    # Get the frame from the server.
                    server.get_frame(client_id, receiver)
                    colour_image = receiver.get_rgb_image()
                    tracker_w_t_c: np.ndarray = receiver.get_pose()

                    # If an output directory has been specified, save the frame to disk.
                    if output_dir is not None:
                        depth_image: np.ndarray = ImageUtil.from_short_depth(receiver.get_depth_image())
                        RGBDSequenceUtil.save_frame(
                            frame_idx, output_dir, colour_image, depth_image, tracker_w_t_c,
                            colour_intrinsics=intrinsics, depth_intrinsics=intrinsics
                        )
                        frame_idx += 1

                    # If the depth estimator hasn't been constructed yet, construct it now.
                    if depth_estimator is None:
                        depth_estimator = MonocularDepthEstimator(
                            "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar",
                            GeometryUtil.intrinsics_to_matrix(intrinsics),
                            debug=True
                        )

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
