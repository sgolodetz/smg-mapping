import cv2
import detectron2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import threading
import time

from detectron2.structures import Instances
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
from smg.mapping.remote import MappingServer, RGBDFrameReceiver
from smg.mvdepthnet import MonocularDepthEstimator
from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.pyoctomap import CM_COLOR_HEIGHT, OctomapUtil, OcTree, OcTreeDrawer, Pointcloud, Vector3
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import GeometryUtil, RGBDSequenceUtil


class MVDepthOctomapMappingSystem:
    """A mapping system that estimates depths using MVDepthNet and reconstructs an Octomap."""

    # CONSTRUCTOR

    def __init__(self, server: MappingServer, depth_estimator: MonocularDepthEstimator, *, camera_mode: str = "free",
                 detect_objects: bool = False, output_dir: Optional[str] = None, save_frames: bool = False,
                 save_reconstruction: bool = False, window_size: Tuple[int, int] = (640, 480)):
        """
        Construct a mapping system that estimates depths using MVDepthNet and reconstructs an Octomap.

        :param server:              The mapping server.
        :param depth_estimator:     The monocular depth estimator.
        :param camera_mode:         The camera mode to use (follow|free).
        :param detect_objects:      Whether to detect 3D objects.
        :param output_dir:          An optional directory into which to save output files.
        :param save_frames:         Whether to save the sequence of frames used to reconstruct the Octomap.
        :param save_reconstruction: Whether to save the reconstructed Octomap.
        :param window_size:         The size of window to use.
        """
        self.__camera_mode: str = camera_mode
        self.__client_id: int = 0
        self.__depth_estimator: MonocularDepthEstimator = depth_estimator
        self.__detect_objects: bool = detect_objects
        self.__output_dir: Optional[str] = output_dir
        self.__save_frames: bool = save_frames
        self.__save_reconstruction: bool = save_reconstruction
        self.__server: MappingServer = server
        self.__should_terminate: threading.Event = threading.Event()
        self.__window_size: Tuple[int, int] = window_size

        # The image size and camera intrinsics, together with their lock.
        self.__image_size: Optional[Tuple[int, int]] = None
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__parameters_lock: threading.Lock = threading.Lock()

        # The detection inputs, together with their lock.
        self.__detection_colour_image: Optional[np.ndarray] = None
        self.__detection_depth_image: Optional[np.ndarray] = None
        self.__detection_w_t_c: Optional[np.ndarray] = None
        self.__detection_lock: threading.Lock = threading.Lock()

        # The most recent instance segmentation, the detected 3D objects and the octree, together with their lock.
        self.__instance_segmentation: Optional[np.ndarray] = None
        self.__objects: List[ObjectDetector3D.Object3D] = []
        self.__octree: Optional[OcTree] = None
        self.__scene_lock: threading.Lock = threading.Lock()

        # The threads and conditions.
        self.__detection_thread: Optional[threading.Thread] = None
        self.__detection_input_is_ready: bool = False
        self.__detection_input_ready: threading.Condition = threading.Condition(self.__detection_lock)

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
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        viewing_pose: np.ndarray = np.eye(4)

        # Initialise PyGame and create the window.
        pygame.init()
        pygame.display.set_mode(self.__window_size, pygame.DOUBLEBUF | pygame.OPENGL)
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

        # If we're detecting 3D objects, start the detection thread.
        if self.__detect_objects:
            self.__detection_thread = threading.Thread(target=self.__run_detection)
            self.__detection_thread.start()

        # Until the mapping system should terminate:
        while not self.__should_terminate.is_set():
            # Process any PyGame events.
            for event in pygame.event.get():
                # If the user wants to quit:
                if event.type == pygame.QUIT:
                    # Shut down pygame, close any remaining OpenCV windows, and exit.
                    pygame.quit()
                    cv2.destroyAllWindows()
                    return

            # Try to get the camera parameters.
            with self.__parameters_lock:
                image_size: Optional[Tuple[int, int]] = self.__image_size
                intrinsics: Optional[Tuple[float, float, float, float]] = self.__intrinsics

            # Show the most recent instance segmentation (if any).
            with self.__scene_lock:
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
                if self.__camera_mode == "follow":
                    if self.__server.peek_newest_frame(self.__client_id, receiver):
                        viewing_pose = np.linalg.inv(receiver.get_pose())
                else:
                    viewing_pose = camera_controller.get_pose()

                # Set the projection matrix.
                with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                    GeometryUtil.rescale_intrinsics(intrinsics, image_size, self.__window_size), *self.__window_size
                )):
                    # Set the model-view matrix.
                    with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                        CameraPoseConverter.pose_to_modelview(viewing_pose)
                    )):
                        # Draw the voxel grid.
                        glColor3f(0.0, 0.0, 0.0)
                        OpenGLUtil.render_voxel_grid([-3, -2, -3], [3, 0, 3], [1, 1, 1], dotted=True)

                        with self.__scene_lock:
                            # Draw the octree.
                            if self.__octree is not None:
                                OctomapUtil.draw_octree(self.__octree, drawer)

                            # Draw the 3D objects.
                            glColor3f(1.0, 0.0, 1.0)
                            for obj in self.__objects:
                                OpenGLUtil.render_aabb(*obj.box_3d)

            # Swap the front and back buffers.
            pygame.display.flip()

    def terminate(self) -> None:
        """Destroy the mapping system."""
        if not self.__should_terminate.is_set():
            self.__should_terminate.set()

            # Join any running threads.
            if self.__detection_thread is not None:
                self.__detection_thread.join()
            if self.__mapping_thread is not None:
                self.__mapping_thread.join()

            # If an output directory has been specified and we're saving the reconstruction, save it now.
            if self.__output_dir is not None and self.__save_reconstruction:
                os.makedirs(self.__output_dir, exist_ok=True)
                output_filename: str = os.path.join(self.__output_dir, "octree.bt")
                print(f"Saving octree to: {output_filename}")
                self.__octree.write_binary(output_filename)

    # PRIVATE METHODS

    def __run_detection(self) -> None:
        """Run the detection thread."""
        # Construct the instance segmenter and object detector.
        instance_segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()
        object_detector: ObjectDetector3D = ObjectDetector3D(instance_segmenter)

        # Until termination is requested:
        while not self.__should_terminate.is_set():
            with self.__detection_lock:
                # Wait for a detection request. If termination is requested whilst waiting, exit.
                while not self.__detection_input_is_ready:
                    self.__detection_input_ready.wait(0.1)
                    if self.__should_terminate.is_set():
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

                # Lift relevant 2D instances to 3D objects.
                # TODO: Ultimately, they should be fused in - this is a first pass.
                instances: List[InstanceSegmenter.Instance] = instance_segmenter.parse_raw_instances(raw_instances)
                instances = [instance for instance in instances if instance.label != "book"]
                objects: List[ObjectDetector3D.Object3D] = object_detector.lift_to_3d(
                    instances, self.__detection_depth_image, self.__detection_w_t_c, intrinsics
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
        frame_idx: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

        # Construct the octree.
        voxel_size: float = 0.05
        self.__octree = OcTree(voxel_size)
        self.__octree.set_occupancy_thres(0.8)

        # Until termination is requested:
        while not self.__should_terminate.is_set():
            # If the server has a frame from the client that has not yet been processed:
            if self.__server.has_frames_now(self.__client_id):
                # Get the camera parameters from the server.
                height, width, _ = self.__server.get_image_shapes(self.__client_id)[0]
                intrinsics: Tuple[float, float, float, float] = self.__server.get_intrinsics(self.__client_id)[0]

                # Record them so that other threads have access to them.
                with self.__parameters_lock:
                    self.__image_size = (width, height)
                    self.__intrinsics = intrinsics

                # Pass the camera intrinsics to the depth estimator.
                self.__depth_estimator.set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                # Get the frame from the server.
                self.__server.get_frame(self.__client_id, receiver)
                colour_image: np.ndarray = receiver.get_rgb_image()
                mapping_w_t_c: np.ndarray = receiver.get_pose()

                # If an output directory was specified and we're saving frames, save the frame to disk.
                if self.__output_dir is not None and self.__save_frames:
                    depth_image: np.ndarray = receiver.get_depth_image()
                    RGBDSequenceUtil.save_frame(
                        frame_idx, self.__output_dir, colour_image, depth_image, mapping_w_t_c,
                        colour_intrinsics=intrinsics, depth_intrinsics=intrinsics
                    )
                    frame_idx += 1

                # Try to estimate a depth image for the frame.
                start = timer()

                estimated_depth_image: Optional[np.ndarray] = self.__depth_estimator.estimate_depth(
                    colour_image, mapping_w_t_c
                )

                end = timer()
                print(f"  - Depth Estimation Time: {end - start}s")

                # If a depth image was successfully estimated:
                if estimated_depth_image is not None:
                    # Limit its range to 3m (more distant points can be unreliable).
                    estimated_depth_image = np.where(estimated_depth_image <= 3.0, estimated_depth_image, 0.0)

                    # Use the depth image and pose to make an Octomap point cloud.
                    pcd: Pointcloud = OctomapUtil.make_point_cloud(estimated_depth_image, mapping_w_t_c, intrinsics)

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
                                self.__detection_w_t_c = mapping_w_t_c.copy()
                                self.__detection_input_is_ready = True
                                self.__detection_input_ready.notify()
                        finally:
                            self.__detection_lock.release()
            else:
                # If no new frame is currently available, wait for 10ms to avoid a spin loop.
                time.sleep(0.01)
