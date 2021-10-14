import cv2
import detectron2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import threading
import time

from detectron2.structures import Instances
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.base import RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.comms.skeletons import RemoteSkeletonDetector
from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
from smg.meshing import MeshUtil
from smg.open3d import ReconstructionUtil
from smg.opengl import OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.pyoctomap import CM_COLOR_HEIGHT, OctomapPicker, OctomapUtil, OcTree, OcTreeDrawer, Pointcloud, Vector3
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Skeleton3D, SkeletonRenderer, SkeletonUtil
from smg.utility import GeometryUtil, ImageUtil, MonocularDepthEstimator, SequenceUtil

from ..selectors.bone_selector import BoneSelector


class OctomapMappingSystem:
    """A mapping system that reconstructs an Octomap."""

    # CONSTRUCTOR

    def __init__(self, server: MappingServer, depth_estimator: MonocularDepthEstimator, *,
                 camera_mode: str = "free", detect_objects: bool = False, detect_skeletons: bool = False,
                 max_received_depth: float = 3.0, output_dir: Optional[str] = None, postprocess_depth: bool = True,
                 save_frames: bool = False, save_reconstruction: bool = False, save_skeletons: bool = False,
                 use_arm_selection: bool = False, use_received_depth: bool = False,
                 window_size: Tuple[int, int] = (640, 480)):
        """
        Construct a mapping system that reconstructs an Octomap.

        :param server:              The mapping server.
        :param depth_estimator:     The monocular depth estimator.
        :param camera_mode:         The camera mode to use (follow|free).
        :param detect_objects:      Whether to detect 3D objects.
        :param detect_skeletons:    Whether to detect 3D skeletons.
        :param max_received_depth:  The maximum depth values to keep when using the received depth (pixels with
                                    depth values greater than this will have their depths set to zero).
        :param output_dir:          An optional directory into which to save output files.
        :param postprocess_depth:   Whether to post-process the depth images.
        :param save_frames:         Whether to save the sequence of frames used to reconstruct the Octomap.
        :param save_reconstruction: Whether to save the reconstructed Octomap.
        :param save_skeletons:      Whether to save the skeletons detected in each frame.
        :param use_arm_selection:   Whether to allow the user to select 3D points in the scene using their arm.
        :param use_received_depth:  Whether to use depth images received from the client instead of estimating depth.
        :param window_size:         The size of window to use.
        """
        self.__camera_mode: str = camera_mode
        self.__client_id: int = 0
        self.__depth_estimator: MonocularDepthEstimator = depth_estimator
        self.__detect_objects: bool = detect_objects
        self.__detect_skeletons: bool = detect_skeletons
        self.__max_received_depth: float = max_received_depth
        self.__output_dir: Optional[str] = output_dir
        self.__postprocess_depth: bool = postprocess_depth
        self.__save_frames: bool = save_frames
        self.__save_reconstruction: bool = save_reconstruction
        self.__save_skeletons: bool = save_skeletons
        self.__server: MappingServer = server
        self.__should_terminate: threading.Event = threading.Event()
        self.__use_arm_selection: bool = use_arm_selection
        self.__use_received_depth: bool = use_received_depth
        self.__window_size: Tuple[int, int] = window_size

        # The image size and camera intrinsics, together with their lock.
        self.__image_size: Optional[Tuple[int, int]] = None
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__parameters_lock: threading.Lock = threading.Lock()

        # The object detection inputs, together with their lock.
        self.__object_detection_colour_image: Optional[np.ndarray] = None
        self.__object_detection_depth_image: Optional[np.ndarray] = None
        self.__object_detection_w_t_c: Optional[np.ndarray] = None
        self.__object_detection_lock: threading.Lock = threading.Lock()

        # The most recent instance segmentation, the detected 3D objects and the octree, together with their lock.
        self.__instance_segmentation: Optional[np.ndarray] = None
        self.__mesh: Optional[OpenGLTriMesh] = None
        self.__mesh_needs_updating: threading.Event = threading.Event()
        self.__objects: List[ObjectDetector3D.Object3D] = []
        self.__octree: Optional[OcTree] = None
        self.__scene_lock: threading.Lock = threading.Lock()
        self.__skeletons: List[Skeleton3D] = []
        self.__tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=0.1,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        self.__visualise_mesh: threading.Event = threading.Event()

        # The threads and conditions.
        self.__mapping_thread: Optional[threading.Thread] = None

        self.__object_detection_thread: Optional[threading.Thread] = None
        self.__object_detection_input_is_ready: bool = False
        self.__object_detection_input_ready: threading.Condition = threading.Condition(self.__object_detection_lock)

        self.__skeleton_detection_thread: Optional[threading.Thread] = None

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
        picker: Optional[OctomapPicker] = None
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        viewing_pose: np.ndarray = np.eye(4)

        # Initialise PyGame and create the window.
        pygame.init()
        pygame.display.set_mode(self.__window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Octomap Mapping Server")

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

        # If we're detecting 3D objects, start the object detection thread.
        if self.__detect_objects:
            self.__object_detection_thread = threading.Thread(target=self.__run_object_detection)
            self.__object_detection_thread.start()

        # If we're detecting 3D skeletons, start the skeleton detection thread.
        if self.__detect_skeletons:
            self.__skeleton_detection_thread = threading.Thread(target=self.__run_skeleton_detection)
            self.__skeleton_detection_thread.start()

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

            pressed_keys = pygame.key.get_pressed()

            if pressed_keys[pygame.K_v]:
                self.__visualise_mesh.set()
            else:
                self.__visualise_mesh.clear()

            # Allow the user to control the camera.
            camera_controller.update(pressed_keys, timer() * 1000)

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

                # Try to let the user select a 3D point in the scene with their arm (if this is enabled).
                selected_point: Optional[np.ndarray] = None

                if self.__use_arm_selection:
                    # Try to construct the picker, if it hasn't been constructed already.
                    if picker is None:
                        with self.__scene_lock:
                            if self.__octree is not None:
                                # noinspection PyTypeChecker
                                picker = OctomapPicker(self.__octree, *image_size, intrinsics)

                    # If the picker has now been constructed:
                    if picker is not None:
                        with self.__scene_lock:
                            # If a single skeleton has currently been detected:
                            if len(self.__skeletons) == 1:
                                # Construct the selector and try to select a 3D scene point.
                                selector: BoneSelector = BoneSelector(self.__skeletons[0], "LElbow", "LWrist")
                                selected_point = selector.get_selected_point(picker)

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
                            # Draw the scene.
                            if self.__visualise_mesh.is_set():
                                if self.__mesh_needs_updating.is_set():
                                    o3d_mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(self.__tsdf)
                                    self.__mesh = MeshUtil.convert_trimesh_to_opengl(o3d_mesh)
                                    self.__mesh_needs_updating.clear()

                                if self.__mesh is not None:
                                    self.__mesh.render()
                            elif self.__octree is not None:
                                OctomapUtil.draw_octree(self.__octree, drawer)

                            # Draw the 3D objects.
                            glColor3f(1.0, 0.0, 1.0)
                            for obj in self.__objects:
                                OpenGLUtil.render_aabb(*obj.box_3d)

                            # Draw the 3D skeletons.
                            with SkeletonRenderer.default_lighting_context():
                                for skeleton in self.__skeletons:
                                    SkeletonRenderer.render_skeleton(skeleton)

                            # Draw any 3D scene point that the user selected.
                            if selected_point is not None:
                                glColor3f(1, 0, 1)
                                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                                OpenGLUtil.render_sphere(selected_point, 0.1, slices=10, stacks=10)
                                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Swap the front and back buffers.
            pygame.display.flip()

    def terminate(self) -> None:
        """Destroy the mapping system."""
        if not self.__should_terminate.is_set():
            self.__should_terminate.set()

            # Join any running threads.
            if self.__mapping_thread is not None:
                self.__mapping_thread.join()
            if self.__object_detection_thread is not None:
                self.__object_detection_thread.join()
            if self.__skeleton_detection_thread is not None:
                self.__skeleton_detection_thread.join()

            # If an output directory has been specified and we're saving the reconstruction, save it now.
            if self.__output_dir is not None and self.__save_reconstruction:
                os.makedirs(self.__output_dir, exist_ok=True)
                output_filename: str = os.path.join(self.__output_dir, "octree.bt")
                print(f"Saving octree to: {output_filename}")
                self.__octree.write_binary(output_filename)

    # PRIVATE METHODS

    def __run_mapping(self) -> None:
        """Run the mapping thread."""
        frame_idx: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        skeleton_detector: Optional[RemoteSkeletonDetector] = RemoteSkeletonDetector() \
            if self.__detect_skeletons else None

        # Construct the octree.
        voxel_size: float = 0.05
        self.__octree = OcTree(voxel_size)
        self.__octree.set_occupancy_thres(0.7)

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
                    SequenceUtil.save_rgbd_frame(
                        frame_idx, self.__output_dir, colour_image, depth_image, mapping_w_t_c,
                        colour_intrinsics=intrinsics, depth_intrinsics=intrinsics
                    )
                    frame_idx += 1

                # If we're detecting skeletons:
                if self.__detect_skeletons:
                    # Set the camera calibration for the skeleton detector.
                    # FIXME: Do this once.
                    skeleton_detector.set_calibration((width, height), intrinsics)

                    # Start trying to detect any skeletons in the colour image. If this fails, skip this frame.
                    if not skeleton_detector.begin_detection(colour_image, mapping_w_t_c):
                        time.sleep(0.01)
                        continue

                # Try to estimate (or otherwise obtain) a depth image for the frame.
                start = timer()

                # noinspection PyUnusedLocal
                estimated_depth_image: Optional[np.ndarray] = None

                # If requested, use the depth image received from the (presumably RGB-D) client.
                if self.__use_received_depth:
                    estimated_depth_image = receiver.get_depth_image()

                    # Limit the depth range (more distant points can be unreliable).
                    estimated_depth_image = np.where(
                        estimated_depth_image <= self.__max_received_depth, estimated_depth_image, 0.0
                    )

                # Otherwise:
                else:
                    # Estimate a depth image using the monocular depth estimator.
                    estimated_depth_image = self.__depth_estimator.estimate_depth(
                        colour_image, mapping_w_t_c, postprocess=self.__postprocess_depth
                    )

                end = timer()
                print(f"  - Depth Estimation Time: {end - start}s")

                # If we're detecting skeletons:
                if self.__detect_skeletons:
                    # Get the skeletons that we asked the detector for, together with their associated people mask.
                    skeletons, people_mask = skeleton_detector.end_detection()

                    # If an output directory has been specified and we're saving the detected skeletons:
                    if self.__output_dir is not None and self.__save_skeletons:
                        # Make sure the output directory exists.
                        os.makedirs(self.__output_dir, exist_ok=True)

                        # Save the detected skeletons into a file in the output directory. Note that we use the
                        # frame index obtained from the mapping client to determine the filename, as the reason
                        # we're saving the skeletons is to compare them with the ground truth ones. This is made
                        # easier if the frame numbers used are the same as the ground truth ones.
                        SkeletonUtil.save_skeletons(
                            os.path.join(self.__output_dir, f"{receiver.get_frame_index()}.skeletons.txt"), skeletons
                        )

                # Otherwise:
                else:
                    # Set the people mask to None, which will cause the subsequent depopulation step to be a no-op.
                    people_mask: Optional[np.ndarray] = None

                # If a depth image was successfully estimated:
                if estimated_depth_image is not None:
                    # Remove any detected people from the depth image.
                    depopulated_depth_image: np.ndarray = estimated_depth_image.copy()
                    if people_mask is not None:
                        depopulated_depth_image = SkeletonUtil.depopulate_depth_image(
                            depopulated_depth_image, people_mask
                        )

                    # Use the depth image and pose to make an Octomap point cloud.
                    pcd: Pointcloud = OctomapUtil.make_point_cloud(depopulated_depth_image, mapping_w_t_c, intrinsics)

                    # Fuse the point cloud into the octree.
                    start = timer()

                    with self.__scene_lock:
                        sensor_origin: Vector3 = Vector3(*mapping_w_t_c[0:3, 3])
                        self.__octree.insert_point_cloud(pcd, sensor_origin, discretize=True)

                        fx, fy, cx, cy = intrinsics
                        o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                            width, height, fx, fy, cx, cy
                        )
                        ReconstructionUtil.integrate_frame(
                            ImageUtil.flip_channels(colour_image), estimated_depth_image, np.linalg.inv(mapping_w_t_c),
                            o3d_intrinsics, self.__tsdf
                        )
                        self.__mesh_needs_updating.set()

                    end = timer()
                    print(f"  - Fusion Time: {end - start}s")

                    # If no frame is currently being processed by the 3D object detector, schedule this one.
                    acquired: bool = self.__object_detection_lock.acquire(blocking=False)
                    if acquired:
                        try:
                            if not self.__object_detection_input_is_ready:
                                self.__object_detection_colour_image = colour_image.copy()
                                self.__object_detection_depth_image = estimated_depth_image.copy()
                                self.__object_detection_w_t_c = mapping_w_t_c.copy()
                                self.__object_detection_input_is_ready = True
                                self.__object_detection_input_ready.notify()
                        finally:
                            self.__object_detection_lock.release()
            else:
                # If no new frame is currently available, wait for 10ms to avoid a spin loop.
                time.sleep(0.01)

    def __run_object_detection(self) -> None:
        """Run the object detection thread."""
        # Construct the instance segmenter and object detector.
        instance_segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()
        object_detector: ObjectDetector3D = ObjectDetector3D(instance_segmenter)

        # Until termination is requested:
        while not self.__should_terminate.is_set():
            with self.__object_detection_lock:
                # Wait for a detection request. If termination is requested whilst waiting, exit.
                while not self.__object_detection_input_is_ready:
                    self.__object_detection_input_ready.wait(0.1)
                    if self.__should_terminate.is_set():
                        return

                start = timer()

                # Find any 2D instances in the detection input image.
                raw_instances: detectron2.structures.Instances = instance_segmenter.segment_raw(
                    self.__object_detection_colour_image
                )

                end = timer()
                print(f"  - Segmentation Time: {end - start}s")

                # Draw the 2D instance segmentation so that it can be shown to the user.
                instance_segmentation: np.ndarray = instance_segmenter.draw_raw_instances(
                    raw_instances, self.__object_detection_colour_image
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
                    instances, self.__object_detection_depth_image, self.__object_detection_w_t_c, intrinsics
                )

                end = timer()
                print(f"  - Lifting Time: {end - start}s")

                with self.__scene_lock:
                    # Share the instance segmentation and the detected 3D objects with other threads.
                    self.__instance_segmentation = instance_segmentation.copy()
                    self.__objects = objects

                # Signal that the detector is now idle.
                self.__object_detection_input_is_ready = False

    def __run_skeleton_detection(self) -> None:
        """Run the (real-time) skeleton detection thread."""
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        skeleton_detector: RemoteSkeletonDetector = RemoteSkeletonDetector(("127.0.0.1", 7853))

        # Until termination is requested:
        while not self.__should_terminate.is_set():
            # If the server has any frames from the client that have not yet been processed, and it's thus
            # possible to get the most recent one:
            if self.__server.peek_newest_frame(self.__client_id, receiver):
                # Detect any skeletons in the most recent frame:
                colour_image: np.ndarray = receiver.get_rgb_image()
                world_from_camera: np.ndarray = receiver.get_pose()
                skeletons, _ = skeleton_detector.detect_skeletons(colour_image, world_from_camera)

                # Make any skeletons that were detected available to other threads.
                with self.__scene_lock:
                    self.__skeletons = skeletons if skeletons is not None else []

            # Otherwise, wait for 10ms to avoid a spin lock.
            else:
                time.sleep(0.01)
