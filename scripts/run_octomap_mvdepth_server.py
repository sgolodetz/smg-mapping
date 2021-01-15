import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from operator import itemgetter
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.mapping.remote import MappingServer, RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.mvdepthnet import MVDepthEstimator
from smg.opengl import OpenGLUtil
from smg.pyoctomap import *
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraUtil
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
    pygame.display.set_caption("Octomap MVDepth Server")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Set up the octree drawer.
    drawer: OcTreeDrawer = OcTreeDrawer()
    drawer.set_color_mode(CM_COLOR_HEIGHT)

    # Create the octree.
    voxel_size: float = 0.025
    tree: OcTree = OcTree(voxel_size)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    with MappingServer(frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message) as server:
        client_id: int = 0
        depth_estimator: Optional[MVDepthEstimator] = None
        keyframes: List[Tuple[np.ndarray, np.ndarray]] = []
        intrinsics: Optional[Tuple[float, float, float, float]] = None
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        tracker_w_t_c: Optional[np.ndarray] = None

        # Start the server.
        server.start()

        frame_idx: int = 0

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

            # If the server has an RGB-D frame from the client that has not yet been processed:
            if server.has_frames_now(client_id):
                # Get the camera intrinsics from the server.
                intrinsics = server.get_intrinsics(client_id)[0]

                # Get the frame from the server.
                server.get_frame(client_id, receiver)
                colour_image: np.ndarray = receiver.get_rgb_image()
                tracker_w_t_c = receiver.get_pose()

                print(tracker_w_t_c)

                # If the depth estimator hasn't been constructed yet, construct it now.
                if depth_estimator is None:
                    depth_estimator = MVDepthEstimator(
                        "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar",
                        GeometryUtil.intrinsics_to_matrix(intrinsics)
                    )

                # FIXME: This next bit is mostly all duplicate code - factor it out.
                # Compute the baselines (in m) and look angles (in degrees) with respect to any existing keyframes.
                baselines: List[float] = []
                look_angles: List[float] = []
                for _, keyframe_w_t_c in keyframes:
                    baselines.append(CameraUtil.compute_baseline_p(tracker_w_t_c, keyframe_w_t_c))
                    look_angles.append(CameraUtil.compute_look_angle_p(tracker_w_t_c, keyframe_w_t_c))

                print(f"Frame {frame_idx}")
                frame_idx += 1
                print(f"Keyframes: {[keyframe_w_t_c for _, keyframe_w_t_c in keyframes]}")
                # print(f"Baselines: {baselines}, Look Angles: {look_angles}")

                # Score all of the keyframes with respect to the current frame.
                scores: List[(int, float)] = []
                smallest_baseline: float = 1000.0
                smallest_look_angle: float = 1000.0

                for i in range(len(keyframes)):
                    smallest_baseline = min(baselines[i], smallest_baseline)
                    smallest_look_angle = min(look_angles[i], smallest_look_angle)

                    if baselines[i] < 0.025 or look_angles[i] > 20.0:
                        # If the baseline's too small, force the score of this keyframe to 0.
                        scores.append((i, 0.0))
                    else:
                        # Otherwise, compute a score as per the Mobile3DRecon paper (but with different parameters).
                        b_m: float = 0.15
                        delta: float = 0.1
                        alpha_m: float = 10.0
                        w_b: float = np.exp(-(baselines[i] - b_m) ** 2 / delta ** 2)
                        w_v: float = max(alpha_m / look_angles[i], 1)
                        scores.append((i, w_b * w_v))

                # print(f"Scores: {scores}")

                # Try to choose up to two keyframes to use together with the current frame to estimate the depth.
                best_depth_image: Optional[np.ndarray] = None
                if len(scores) >= 2:
                    # Find the two best keyframes, based on their scores.
                    # FIXME: There's no need to fully sort the list here.
                    scores = sorted(scores, key=itemgetter(1), reverse=True)
                    best_keyframe_idx, best_keyframe_score = scores[0]
                    second_best_keyframe_idx, second_best_keyframe_score = scores[1]
                    print(f"-  {best_keyframe_score}, {second_best_keyframe_score}")

                    # If both keyframes are fine to use:
                    if best_keyframe_score > 0.0 and second_best_keyframe_score > 0.0:
                        # Look up the keyframe images and poses.
                        best_keyframe_image, best_keyframe_w_t_c = keyframes[best_keyframe_idx]
                        second_best_keyframe_image, second_best_keyframe_w_t_c = keyframes[second_best_keyframe_idx]

                        # Separately estimate a depth image from each keyframe.
                        best_depth_image = depth_estimator.estimate_depth(
                            colour_image, best_keyframe_image, tracker_w_t_c, best_keyframe_w_t_c
                        )
                        second_best_depth_image: np.ndarray = depth_estimator.estimate_depth(
                            colour_image, second_best_keyframe_image, tracker_w_t_c, second_best_keyframe_w_t_c
                        )

                        # Filter out any depths that are not sufficiently consistent across both estimates.
                        tolerance: float = 0.1
                        diff: np.ndarray = np.abs(best_depth_image - second_best_depth_image)
                        best_depth_image = np.where(diff < tolerance, best_depth_image, 0.0)
                        # second_best_depth_image = np.where(diff < tolerance, second_best_depth_image, 0.0)

                        # # Show both estimates, after the filtering.
                        # import cv2
                        # cv2.imshow("Best Depth Image", best_depth_image / 2)
                        # cv2.imshow("Second Best Depth Image", second_best_depth_image / 2)
                        # cv2.waitKey(1)

                # Check whether this frame should be a new keyframe. If so, add it to the list.
                # print(f"SB: {smallest_baseline}, SLA: {smallest_look_angle}")
                if smallest_baseline > 0.05 or smallest_look_angle > 5.0:
                    keyframes.append((colour_image.copy(), tracker_w_t_c.copy()))

                # TODO: Comment here.
                if best_depth_image is not None:
                    # TODO
                    best_depth_image = np.where(best_depth_image <= 3.0, best_depth_image, 0.0)

                    # Use the depth image and pose to make an Octomap point cloud.
                    pcd: Pointcloud = OctomapUtil.make_point_cloud(best_depth_image, tracker_w_t_c, intrinsics)

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

            # Once the pose is available:
            if tracker_w_t_c is not None:
                # Set the projection matrix.
                glMatrixMode(GL_PROJECTION)
                OpenGLUtil.set_projection_matrix(intrinsics, *window_size)

                # Draw the octree.
                viewing_pose: np.ndarray = \
                    np.linalg.inv(tracker_w_t_c) if args["camera_mode"] == "follow" \
                    else camera_controller.get_pose()
                OctomapUtil.draw_octree(tree, viewing_pose, drawer)

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
