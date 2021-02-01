import cv2
import detectron2
import numpy as np
import os
import time

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import threading

from detectron2.structures import Instances
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
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
        self.__output_dir: Optional[str] = output_dir
        self.__server: MappingServer = server
        self.__should_terminate: bool = False

        # The image size and camera intrinsics, together with their lock.
        self.__image_size: Optional[Tuple[int, int]] = None
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__parameters_lock: threading.Lock = threading.Lock()

        # The detection inputs/outputs, together with their lock.
        self.__detection_colour_image: Optional[np.ndarray] = None
        self.__detection_depth_image: Optional[np.ndarray] = None
        self.__detection_tracker_w_t_c: Optional[np.ndarray] = None
        self.__detection_lock: threading.Lock = threading.Lock()

        # The mapping pose, the detected 3D objects, the most recent instance segmentation and the octree,
        # together with their lock.
        self.__instance_segmentation: Optional[np.ndarray] = None
        self.__mapping_w_t_c: Optional[np.ndarray] = None
        self.__objects: List[ObjectDetector3D.Object3D] = []
        self.__octree: Optional[OcTree] = None
        self.__scene_lock: threading.Lock = threading.Lock()

        # The threads and conditions.
        self.__detection_thread: Optional[threading.Thread] = None
        self.__detection_input_ready: threading.Condition = threading.Condition(self.__detection_lock)
        self.__detection_input_is_ready: bool = False

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

        # Construct the camera controller.
        camera_controller: KeyboardCameraController = KeyboardCameraController(
            SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
        )

        # Start the mapping thread.
        self.__mapping_thread = threading.Thread(target=self.__run_mapping)
        self.__mapping_thread.start()

        # Start the detection thread.
        self.__detection_thread = threading.Thread(target=self.__run_detection)
        self.__detection_thread.start()

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

            with self.__parameters_lock:
                # Try to get the camera parameters.
                image_size: Optional[Tuple[int, int]] = self.__image_size
                intrinsics: Optional[Tuple[float, float, float, float]] = self.__intrinsics

            with self.__scene_lock:
                # Try to get the mapping pose.
                mapping_w_t_c: Optional[np.ndarray] = self.__mapping_w_t_c.copy() \
                    if self.__mapping_w_t_c is not None else None

                # Show the most recent instance segmentation (if any).
                if self.__instance_segmentation is not None:
                    cv2.imshow("Instance Segmentation", self.__instance_segmentation)
                    cv2.waitKey(1)

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Once at least one frame has been received:
            if image_size is not None:
                # Determine the viewing pose.
                viewing_pose: np.ndarray = \
                    np.linalg.inv(mapping_w_t_c) if self.__camera_mode == "follow" and mapping_w_t_c is not None \
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
                        with self.__scene_lock:
                            if self.__octree is not None:
                                OctomapUtil.draw_octree(self.__octree, drawer)

            # Swap the front and back buffers.
            pygame.display.flip()

    def terminate(self) -> None:
        """Destroy the mapping system."""
        if not self.__should_terminate:
            # TODO: Comment here.
            self.__should_terminate = True

            # Join any running threads.
            if self.__detection_thread is not None:
                self.__detection_thread.join()
            if self.__mapping_thread is not None:
                self.__mapping_thread.join()

    # PRIVATE METHODS

    def __run_detection(self) -> None:
        """Run the detection thread."""
        # Construct the instance segmenter and object detector.
        instance_segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()
        object_detector: ObjectDetector3D = ObjectDetector3D(instance_segmenter)

        # Until termination is requested:
        while not self.__should_terminate:
            with self.__detection_lock:
                # Wait for a detection request. If termination is requested whilst waiting, exit.
                while not self.__detection_input_is_ready:
                    self.__detection_input_ready.wait(0.1)
                    if self.__should_terminate:
                        return

                start = timer()

                # Find any 2D instances in the detection input image.
                raw_instances: detectron2.structures.Instances = instance_segmenter.segment_raw(
                    self.__detection_colour_image
                )

                end = timer()
                print(f"  - Segmentation Time: {end - start}s")

                # Draw the 2D instance segmentation so that it can be shown to the user.
                instance_segmentation: np.ndarray = instance_segmenter.draw_raw_instances(
                    raw_instances, self.__detection_colour_image
                )

                # Get the camera intrinsics.
                with self.__parameters_lock:
                    intrinsics: Tuple[float, float, float, float] = self.__intrinsics

                start = timer()

                # Lift the 2D instances to 3D objects.
                # TODO: Ultimately, they should be fused in - this is a first pass.
                instances: List[InstanceSegmenter.Instance] = instance_segmenter.parse_raw_instances(raw_instances)
                instances = [instance for instance in instances if instance.label != "book"]
                objects: List[ObjectDetector3D.Object3D] = object_detector.lift_to_3d(
                    instances, self.__detection_depth_image, self.__detection_tracker_w_t_c, intrinsics
                )

                end = timer()
                print(f"  - Lifting Time: {end - start}s")

                with self.__scene_lock:
                    # Add the detected 3D objects to the overall list.
                    self.__objects += objects

                    # Share the instance segmentation with other threads.
                    self.__instance_segmentation = instance_segmentation.copy()

                # Signal that the detector is now idle.
                self.__detection_input_is_ready = False

    def __run_mapping(self) -> None:
        """Run the mapping thread."""
        client_id: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

        # Construct the octree.
        voxel_size: float = 0.05
        self.__octree = OcTree(voxel_size)
        self.__octree.set_occupancy_thres(0.8)

        # Until termination is requested:
        while not self.__should_terminate:
            # If the server has a frame from the client that has not yet been processed:
            if self.__server.has_frames_now(client_id):
                # Get the camera parameters from the server.
                height, width, _ = self.__server.get_image_shapes(client_id)[0]
                intrinsics: Tuple[float, float, float, float] = self.__server.get_intrinsics(client_id)[0]

                # Record them so that other threads have access to them.
                with self.__parameters_lock:
                    self.__image_size = (width, height)
                    self.__intrinsics = intrinsics

                # Pass the camera intrinsics to the depth estimator.
                self.__depth_estimator.set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                # Get the frame from the server.
                self.__server.get_frame(client_id, receiver)
                colour_image: np.ndarray = receiver.get_rgb_image()
                tracker_w_t_c: np.ndarray = receiver.get_pose()

                # Record the mapping pose so that other threads have access to it.
                with self.__scene_lock:
                    self.__mapping_w_t_c = tracker_w_t_c.copy()

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

                    with self.__scene_lock:
                        origin: Vector3 = Vector3(0.0, 0.0, 0.0)
                        self.__octree.insert_point_cloud(pcd, origin, discretize=True)

                    end = timer()
                    print(f"  - Fusion Time: {end - start}s")

                    # If no frame is currently being processed by the 3D object detector, schedule this one.
                    acquired: bool = self.__detection_lock.acquire(blocking=False)
                    if acquired:
                        try:
                            if not self.__detection_input_is_ready:
                                self.__detection_colour_image = colour_image.copy()
                                self.__detection_depth_image = estimated_depth_image.copy()
                                self.__detection_tracker_w_t_c = tracker_w_t_c.copy()
                                self.__detection_input_is_ready = True
                                self.__detection_input_ready.notify()
                        finally:
                            self.__detection_lock.release()
            else:
                # If no new frame is currently available, wait for 10ms to avoid a spin loop.
                time.sleep(0.01)
