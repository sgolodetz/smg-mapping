import cv2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import threading
import time

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.base import RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.comms.skeletons import RemoteSkeletonDetector
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

try:
    # noinspection PyUnresolvedReferences
    import detectron2
    # noinspection PyUnresolvedReferences
    from detectron2.structures import Instances
    from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
except ImportError:
    class InstanceSegmenter:
        pass

    class ObjectDetector3D:
        class Object3D:
            pass

try:
    from smg.smplx import SMPLBody
except ImportError:
    class SMPLBody:
        pass


class OctomapMappingSystem:
    """A mapping system that reconstructs an Octomap."""

    # CONSTRUCTOR

    def __init__(self, server: MappingServer, depth_estimator: MonocularDepthEstimator, *,
                 camera_mode: str = "free", detect_objects: bool = False, detect_skeletons: bool = False,
                 max_received_depth: float = 3.0, output_dir: Optional[str] = None, postprocess_depth: bool = True,
                 render_bodies: bool = False, save_frames: bool = False, save_reconstruction: bool = False,
                 save_skeletons: bool = False, use_arm_selection: bool = False, use_received_depth: bool = False,
                 use_tsdf: bool = False, window_size: Tuple[int, int] = (640, 480)):
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
        :param render_bodies:       Whether to render an SMPL body in place of each detected skeleton.
        :param save_frames:         Whether to save the sequence of frames used to reconstruct the Octomap.
        :param save_reconstruction: Whether to save the reconstructed Octomap.
        :param save_skeletons:      Whether to save the skeletons detected in each frame.
        :param use_arm_selection:   Whether to allow the user to select 3D points in the scene using their arm.
        :param use_received_depth:  Whether to use depth images received from the client instead of estimating depth.
        :param use_tsdf:            Whether to reconstruct a TSDF as well as an Octomap (for visualisation purposes).
        :param window_size:         The size of window to use.
        """
        self.__body: Optional[SMPLBody] = None
        self.__camera_mode: str = camera_mode
        self.__client_id: int = 0
        self.__depth_estimator: MonocularDepthEstimator = depth_estimator
        self.__detect_objects: bool = detect_objects
        self.__detect_skeletons: bool = detect_skeletons
        self.__max_received_depth: float = max_received_depth
        self.__output_dir: Optional[str] = output_dir
        self.__postprocess_depth: bool = postprocess_depth
        self.__render_bodies: bool = render_bodies
        self.__save_frames: bool = save_frames
        self.__save_reconstruction: bool = save_reconstruction
        self.__save_skeletons: bool = save_skeletons
        self.__server: MappingServer = server
        self.__should_terminate: threading.Event = threading.Event()
        self.__use_arm_selection: bool = use_arm_selection
        self.__use_received_depth: bool = use_received_depth
        self.__use_tsdf: bool = use_tsdf
        self.__window_size: Tuple[int, int] = window_size

        # The camera parameters, together with their lock and condition.
        self.__image_size: Optional[Tuple[int, int]] = None
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__parameters_lock: threading.Lock = threading.Lock()
        self.__parameters_available: threading.Condition = threading.Condition(self.__parameters_lock)

        # The object detection inputs, together with their lock.
        self.__object_detection_colour_image: Optional[np.ndarray] = None
        self.__object_detection_depth_image: Optional[np.ndarray] = None
        self.__object_detection_w_t_c: Optional[np.ndarray] = None
        self.__object_detection_lock: threading.Lock = threading.Lock()

        # The most recent instance segmentation, the detected 3D objects and the scene representations,
        # together with their lock.
        self.__instance_segmentation: Optional[np.ndarray] = None
        self.__mesh: Optional[OpenGLTriMesh] = None
        self.__mesh_needs_updating: threading.Event = threading.Event()
        self.__objects: List[ObjectDetector3D.Object3D] = []
        self.__octree: Optional[OcTree] = None
        self.__scene_lock: threading.Lock = threading.Lock()
        self.__skeletons: List[Skeleton3D] = []
        self.__tsdf: Optional[o3d.pipelines.integration.ScalableTSDFVolume] = None

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

        # If we're rendering an SMPL body for each skeleton, load in the default body model.
        if self.__render_bodies:
            # FIXME: These paths shouldn't be hard-coded like this.
            self.__body = SMPLBody(
                "male",
                texture_coords_filename="D:/smplx/textures/smpl/texture_coords.npy",
                texture_image_filename="D:/smplx/textures/smpl/surreal/nongrey_male_0170.jpg"
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
                # If the user wants to quit, do so.
                if event.type == pygame.QUIT:
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

            # Get the set of currently pressed keys.
            pressed_keys = pygame.key.get_pressed()

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
                            # If TSDF reconstruction is enabled and 'v' is pressed:
                            if self.__use_tsdf and pressed_keys[pygame.K_v]:
                                # Update the mesh if the TSDF has changed since we last visualised it.
                                if self.__mesh_needs_updating.is_set():
                                    o3d_mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(self.__tsdf)
                                    self.__mesh = MeshUtil.convert_trimesh_to_opengl(o3d_mesh)
                                    self.__mesh_needs_updating.clear()

                                # If the mesh is available, render it.
                                if self.__mesh is not None:
                                    self.__mesh.render()

                            # Otherwise, if the octree is available, render it.
                            elif self.__octree is not None:
                                OctomapUtil.draw_octree(self.__octree, drawer)

                            # Draw the 3D objects.
                            glColor3f(1.0, 0.0, 1.0)
                            for obj in self.__objects:
                                OpenGLUtil.render_aabb(*obj.box_3d)

                            # Draw the detected people.
                            with SkeletonRenderer.default_lighting_context():
                                for skeleton in self.__skeletons:
                                    if self.__render_bodies:
                                        self.__body.render_from_skeleton(skeleton)
                                    else:
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

            # If the mesh was ever created, destroy it prior to shutting down pygame (essential to prevent errors).
            del self.__mesh

            # Shut down pygame and close any remaining OpenCV windows.
            pygame.quit()
            cv2.destroyAllWindows()

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
        voxel_size: float = 0.1
        self.__octree = OcTree(voxel_size)
        self.__octree.set_occupancy_thres(0.7)

        # If requested, also construct the TSDF.
        if self.__use_tsdf:
            self.__tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.05,
                sdf_trunc=0.2,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

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
                    self.__parameters_available.notify_all()

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
                    if not skeleton_detector.begin_detection(
                        colour_image, mapping_w_t_c, frame_idx=receiver.get_frame_index()
                    ):
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

                    end = timer()
                    print(f"  - Fusion Time: {end - start}s")

                    # If requested, also fuse the RGB-D image into the TSDF.
                    if self.__use_tsdf:
                        fx, fy, cx, cy = intrinsics
                        o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                            width, height, fx, fy, cx, cy
                        )
                        ReconstructionUtil.integrate_frame(
                            ImageUtil.flip_channels(colour_image), depopulated_depth_image,
                            np.linalg.inv(mapping_w_t_c), o3d_intrinsics, self.__tsdf,
                            depth_trunc=self.__max_received_depth
                        )
                        self.__mesh_needs_updating.set()

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
        # Get the camera intrinsics.
        intrinsics: Optional[Tuple[float, float, float, float]] = None
        intrinsics_sent: bool = False
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        skeleton_detector: RemoteSkeletonDetector = RemoteSkeletonDetector(("127.0.0.1", 7853))

        # Until termination is requested:
        while not self.__should_terminate.is_set():
            # If the camera intrinsics haven't yet been received:
            if intrinsics is None:
                # Wait for them to be received. If termination is requested in the meantime, early out.
                with self.__parameters_lock:
                    while self.__intrinsics is None:
                        self.__parameters_available.wait(0.1)
                        if self.__should_terminate.is_set():
                            return

                    intrinsics = self.__intrinsics

            # If the server has any frames from the client that have not yet been processed, and it's thus
            # possible to get the most recent one:
            if self.__server.peek_newest_frame(self.__client_id, receiver):
                # Get the most recent frame.
                colour_image: np.ndarray = receiver.get_rgb_image()
                frame_idx: int = receiver.get_frame_index()
                world_from_camera: np.ndarray = receiver.get_pose()

                # Send across the camera parameters if necessary.
                if not intrinsics_sent:
                    skeleton_detector.set_calibration(colour_image.shape[:2], intrinsics)
                    intrinsics_sent = True

                # Detect any skeletons in the most recent frame.
                skeletons, _ = skeleton_detector.detect_skeletons(colour_image, world_from_camera, frame_idx=frame_idx)

                # Make any skeletons that were detected available to other threads.
                with self.__scene_lock:
                    self.__skeletons = skeletons if skeletons is not None else []

            # Otherwise, wait for 10ms to avoid a spin lock.
            else:
                time.sleep(0.01)
