import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Optional, Tuple

from smg.mapping.remote import MappingServer, RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.mvdepthnet import MonocularDepthEstimator
from smg.opengl import OpenGLUtil
from smg.pyoctomap import *
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.utility import GeometryUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--camera_mode", "-m", type=str, choices=("follow", "free"), default="free",
        help="the camera mode"
    )
    args: dict = vars(parser.parse_args())

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("MVDepth -> Octomap Server")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Set up the octree drawer.
    drawer: OcTreeDrawer = OcTreeDrawer()
    drawer.set_color_mode(CM_COLOR_HEIGHT)

    # Create the octree.
    voxel_size: float = 0.05
    tree: OcTree = OcTree(voxel_size)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the mapping server.
    with MappingServer(frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message) as server:
        client_id: int = 0
        depth_estimator: Optional[MonocularDepthEstimator] = None
        image_size: Optional[Tuple[int, int]] = None
        intrinsics: Optional[Tuple[float, float, float, float]] = None
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        tracker_w_t_c: Optional[np.ndarray] = None

        # Start the server.
        server.start()

        while True:
            # Process any PyGame events.
            for event in pygame.event.get():
                # If the user wants to quit:
                if event.type == pygame.QUIT:
                    # If the reconstruction process has actually started:
                    if tracker_w_t_c is not None:
                        # Save the current octree to disk.
                        print("Saving octree to remote_fusion_mvdepth.bt")
                        tree.write_binary("remote_fusion_mvdepth.bt")

                    # Shut down pygame, and forcibly exit the program.
                    pygame.quit()
                    # noinspection PyProtectedMember
                    os._exit(0)

            # If the server has a frame from the client that has not yet been processed:
            if server.has_frames_now(client_id):
                # Get the camera parameters from the server.
                height, width, _ = server.get_image_shapes(client_id)[0]
                image_size = (width, height)
                intrinsics = server.get_intrinsics(client_id)[0]

                # Get the frame from the server.
                server.get_frame(client_id, receiver)
                colour_image: np.ndarray = receiver.get_rgb_image()
                tracker_w_t_c = receiver.get_pose()

                # If the depth estimator hasn't been constructed yet, construct it now.
                if depth_estimator is None:
                    depth_estimator = MonocularDepthEstimator(
                        "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"
                    ).set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                # Try to estimate a depth image for the frame.
                estimated_depth_image: Optional[np.ndarray] = depth_estimator.estimate_depth(
                    colour_image, tracker_w_t_c
                )

                # If a depth image was successfully estimated:
                if estimated_depth_image is not None:
                    # Limit its range to 3m (more distant points can be unreliable).
                    estimated_depth_image = np.where(estimated_depth_image <= 3.0, estimated_depth_image, 0.0)

                    # Use the depth image and pose to make an Octomap point cloud.
                    pcd: Pointcloud = OctomapUtil.make_point_cloud(estimated_depth_image, tracker_w_t_c, intrinsics)

                    # Fuse the point cloud into the octree.
                    start = timer()

                    origin: Vector3 = Vector3(0.0, 0.0, 0.0)
                    tree.insert_point_cloud(pcd, origin, discretize=True)

                    end = timer()
                    print(f"  - Time: {end - start}s")

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Once at least one frame has been received:
            if image_size is not None:
                # Set the projection matrix.
                glMatrixMode(GL_PROJECTION)
                rescaled_intrinsics: Tuple[float, float, float, float] = GeometryUtil.rescale_intrinsics(
                    intrinsics, image_size, window_size
                )
                OpenGLUtil.set_projection_matrix(rescaled_intrinsics, *window_size)

                # Draw the octree.
                viewing_pose: np.ndarray = \
                    np.linalg.inv(tracker_w_t_c) if args["camera_mode"] == "follow" and tracker_w_t_c is not None \
                    else camera_controller.get_pose()
                OctomapUtil.draw_octree(tree, viewing_pose, drawer)

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
