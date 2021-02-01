import numpy as np
import os
import time

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import threading

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.mapping.remote import MappingServer, RGBDFrameReceiver
from smg.mvdepthnet import MonocularDepthEstimator
from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.pyoctomap import *
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import GeometryUtil


class MVDepthOctomapMappingSystem:
    """A mapping system that estimates depths using MVDepthNet and reconstructs an Octomap."""

    # CONSTRUCTOR

    def __init__(self, *, depth_estimator: MonocularDepthEstimator, output_dir: Optional[str] = None,
                 server: MappingServer):
        """
        Construct a mapping system that estimates depths using MVDepthNet and reconstructs an Octomap.

        :param depth_estimator: The monocular depth estimator.
        :param output_dir:      TODO
        :param server:          The mapping server.
        """
        self.__camera_mode: str = "free"
        self.__depth_estimator: MonocularDepthEstimator = depth_estimator
        self.__image_size: Optional[Tuple[int, int]] = None
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__output_dir: Optional[str] = output_dir
        self.__server: MappingServer = server
        self.__should_terminate: bool = False
        self.__tracker_w_t_c: Optional[np.ndarray] = None
        self.__tree: Optional[OcTree] = None

        self.__mapping_lock: threading.Lock = threading.Lock()
        self.__mapping_thread: Optional[threading.Thread] = None

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the mapping system's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the mapping system at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the mapping system."""
        # Initialise PyGame and create the window.
        pygame.init()
        window_size: Tuple[int, int] = (640, 480)
        pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("MVDepth -> Octomap Mapping System")

        # Enable the z-buffer.
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Set up the octree drawer.
        drawer: OcTreeDrawer = OcTreeDrawer()
        drawer.set_color_mode(CM_COLOR_HEIGHT)

        # Create the octree.
        voxel_size: float = 0.05
        self.__tree = OcTree(voxel_size)
        self.__tree.set_occupancy_thres(0.8)

        # Construct the camera controller.
        camera_controller: KeyboardCameraController = KeyboardCameraController(
            SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
        )

        # Start the mapping thread.
        self.__mapping_thread = threading.Thread(target=self.__run_mapping)
        self.__mapping_thread.start()

        # Until the mapping system should terminate:
        while not self.__should_terminate:
            # Process any PyGame events.
            for event in pygame.event.get():
                # If the user wants to quit:
                if event.type == pygame.QUIT:
                    # Shut down pygame, and forcibly exit the program.
                    pygame.quit()
                    # noinspection PyProtectedMember
                    os._exit(0)

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Once at least one frame has been received:
            with self.__mapping_lock:
                image_size: Optional[Tuple[int, int]] = self.__image_size
                intrinsics: Optional[Tuple[float, float, float, float]] = self.__intrinsics
                tracker_w_t_c: Optional[np.ndarray] = self.__tracker_w_t_c.copy() \
                    if self.__tracker_w_t_c is not None else None

            if image_size is not None:
                # Determine the viewing pose.
                viewing_pose: np.ndarray = \
                    np.linalg.inv(tracker_w_t_c) if self.__camera_mode == "follow" and tracker_w_t_c is not None \
                    else camera_controller.get_pose()

                # Set the projection matrix.
                with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                    GeometryUtil.rescale_intrinsics(intrinsics, image_size, window_size), *window_size
                )):
                    # Set the model-view matrix.
                    with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                        CameraPoseConverter.pose_to_modelview(viewing_pose)
                    )):
                        # Draw the voxel grid.
                        glColor3f(0.0, 0.0, 0.0)
                        OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                        # Draw the octree.
                        with self.__mapping_lock:
                            OctomapUtil.draw_octree(self.__tree, drawer)

            # Swap the front and back buffers.
            pygame.display.flip()

    def terminate(self) -> None:
        """Destroy the mapping system."""
        if not self.__should_terminate:
            # TODO: Comment here.
            self.__should_terminate = True

            # TODO: Comment here.
            if self.__mapping_thread is not None:
                self.__mapping_thread.join()

    # PRIVATE METHODS

    def __run_mapping(self) -> None:
        """Run the mapping thread."""
        client_id: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

        # Until termination is requested:
        while not self.__should_terminate:
            # If the server has a frame from the client that has not yet been processed:
            if self.__server.has_frames_now(client_id):
                # Get the camera parameters from the server.
                height, width, _ = self.__server.get_image_shapes(client_id)[0]
                intrinsics: Tuple[float, float, float, float] = self.__server.get_intrinsics(client_id)[0]

                # Record them so that the main thread has access to them.
                with self.__mapping_lock:
                    self.__image_size = (width, height)
                    self.__intrinsics = intrinsics

                # Pass the camera intrinsics to the depth estimator.
                self.__depth_estimator.set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                # Get the frame from the server.
                self.__server.get_frame(client_id, receiver)
                colour_image: np.ndarray = receiver.get_rgb_image()
                tracker_w_t_c: np.ndarray = receiver.get_pose()

                # Record the pose so that the main thread has access to it.
                with self.__mapping_lock:
                    self.__tracker_w_t_c = tracker_w_t_c.copy()

                # Try to estimate a depth image for the frame.
                estimated_depth_image: Optional[np.ndarray] = self.__depth_estimator.estimate_depth(
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

                    with self.__mapping_lock:
                        origin: Vector3 = Vector3(0.0, 0.0, 0.0)
                        self.__tree.insert_point_cloud(pcd, origin, discretize=True)

                    end = timer()
                    print(f"  - Time: {end - start}s")
            else:
                # If no new frame is currently available, wait for 10ms to avoid a spin loop.
                time.sleep(0.01)
