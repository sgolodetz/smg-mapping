import cv2
import numpy as np
import open3d as o3d
import threading
import time

from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.base import RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.comms.skeletons import RemoteSkeletonDetector
from smg.open3d import ReconstructionUtil
from smg.relocalisation import ArUcoPnPRelocaliser
from smg.skeletons import SkeletonUtil
from smg.utility import GeometryUtil, ImageUtil, MonocularDepthEstimator, SequenceUtil

try:
    # noinspection PyUnresolvedReferences
    import detectron2
    from detectron2.structures import Instances
    from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
except ImportError:
    class InstanceSegmenter:
        pass

    class ObjectDetector3D:
        class Object3D:
            pass


class Open3DMappingSystem:
    """A mapping system that reconstructs an Open3D TSDF."""

    # CONSTRUCTOR

    def __init__(self, server: MappingServer, depth_estimator: MonocularDepthEstimator, *,
                 aruco_relocaliser: Optional[ArUcoPnPRelocaliser] = None, batch_mode: bool = False,
                 debug: bool = False, detect_objects: bool = False, detect_skeletons: bool = False,
                 max_received_depth: float = 3.0, output_dir: Optional[str] = None, postprocess_depth: bool = True,
                 save_frames: bool = False, use_received_depth: bool = False):
        """
        Construct a mapping system that reconstructs an Open3D TSDF.

        :param server:              The mapping server.
        :param depth_estimator:     The monocular depth estimator.
        :param aruco_relocaliser:   An optional ArUco+PnP relocaliser that can be used to align the map with a marker.
        :param batch_mode:          Whether to use batch mode.
        :param debug:               Whether to enable debugging.
        :param detect_objects:      Whether to detect 3D objects.
        :param detect_skeletons:    Whether to detect 3D skeletons.
        :param max_received_depth:  The maximum depth values to keep when using the received depth (pixels with
                                    depth values greater than this will have their depths set to zero).
        :param output_dir:          An optional directory into which to save output files.
        :param postprocess_depth:   Whether to post-process the depth images.
        :param save_frames:         Whether to save the sequence of frames used to reconstruct the TSDF.
        :param use_received_depth:  Whether to use depth images received from the client instead of estimating depth.
        """
        self.__aruco_from_world_estimates: List[np.ndarray] = []
        self.__aruco_relocaliser: Optional[ArUcoPnPRelocaliser] = aruco_relocaliser
        self.__batch_mode: bool = batch_mode
        self.__client_id: int = 0
        self.__debug: bool = debug
        self.__depth_estimator: MonocularDepthEstimator = depth_estimator
        self.__detect_objects: bool = detect_objects
        self.__detect_skeletons: bool = detect_skeletons
        self.__max_received_depth: float = max_received_depth
        self.__output_dir: Optional[str] = output_dir
        self.__postprocess_depth: bool = postprocess_depth
        self.__save_frames: bool = save_frames
        self.__server: MappingServer = server
        self.__should_terminate: threading.Event = threading.Event()
        self.__use_received_depth: bool = use_received_depth

        # The image size and camera intrinsics, together with their lock.
        self.__calibration_lock: threading.Lock = threading.Lock()
        self.__image_size: Optional[Tuple[int, int]] = None
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None

        # The detection inputs, together with their lock.
        self.__detection_colour_image: Optional[np.ndarray] = None
        self.__detection_depth_image: Optional[np.ndarray] = None
        self.__detection_w_t_c: Optional[np.ndarray] = None
        self.__detection_lock: threading.Lock = threading.Lock()

        # The most recent instance segmentation, the detected 3D objects and the TSDF, together with their lock.
        self.__instance_segmentation: Optional[np.ndarray] = None
        self.__objects: List[ObjectDetector3D.Object3D] = []
        self.__tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=0.1,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        self.__scene_lock: threading.Lock = threading.Lock()

        # The threads and conditions.
        self.__detection_thread: Optional[threading.Thread] = None
        self.__detection_input_is_ready: bool = False
        self.__detection_input_ready: threading.Condition = threading.Condition(self.__detection_lock)

        self.__mapping_thread: Optional[threading.Thread] = None

    # PUBLIC METHODS

    def get_aruco_from_world(self) -> Optional[np.ndarray]:
        """
        Try to get the transformation from world space to ArUco marker space.

        .. note::
            This will only be available if we're aligning the map with a marker using an ArUco+PnP relocaliser.

        :return:    The transformation from world space to ArUco marker space, if available, or None otherwise.
        """
        if len(self.__aruco_from_world_estimates) > 0:
            return GeometryUtil.blend_rigid_transforms(self.__aruco_from_world_estimates)
        else:
            return None

    def run(self) -> Tuple[o3d.pipelines.integration.ScalableTSDFVolume, List[ObjectDetector3D.Object3D]]:
        """
        Run the mapping system.

        :return:    The results of the reconstruction process, as a (TSDF, list of 3D objects) tuple.
        """
        newest_colour_image: Optional[np.ndarray] = None
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

        # Start the mapping thread.
        self.__mapping_thread = threading.Thread(target=self.__run_mapping)
        self.__mapping_thread.start()

        # If we're detecting 3D objects, start the detection thread.
        if self.__detect_objects:
            self.__detection_thread = threading.Thread(target=self.__run_detection)
            self.__detection_thread.start()

        # Until the mapping system should terminate:
        while not self.__should_terminate.is_set():
            # If we're in batch mode and the server will never receive any more frames from the client, exit.
            if self.__batch_mode and self.__server.has_finished(self.__client_id):
                return self.terminate()

            # If the server has ever received a frame from the client, get the colour image from the most recent one.
            if self.__server.peek_newest_frame(self.__client_id, receiver):
                newest_colour_image = receiver.get_rgb_image()

            # If we've ever seen a frame:
            if newest_colour_image is not None:
                # Show the most recent colour image.
                cv2.imshow("Open3D Mapping Server", newest_colour_image)
                c: int = cv2.waitKey(1)

                # If the user presses 'v', exit.
                if c == ord('v'):
                    return self.terminate()

            # Show the most recent instance segmentation (if any).
            with self.__scene_lock:
                if self.__instance_segmentation is not None:
                    cv2.imshow("Instance Segmentation", self.__instance_segmentation)
                    cv2.waitKey(1)

    def terminate(self) -> Tuple[o3d.pipelines.integration.ScalableTSDFVolume, List[ObjectDetector3D.Object3D]]:
        """
        Destroy the mapping system.

        :return:    The results of the reconstruction process, as a (TSDF, list of 3D objects) tuple.
        """
        if not self.__should_terminate.is_set():
            self.__should_terminate.set()

            # Join any running threads.
            if self.__detection_thread is not None:
                self.__detection_thread.join()
            if self.__mapping_thread is not None:
                self.__mapping_thread.join()

        return self.__tsdf, self.__objects

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
                with self.__calibration_lock:
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
                    # Share the instance segmentation and the detected 3D objects with other threads.
                    self.__instance_segmentation = instance_segmentation.copy()
                    self.__objects = objects

                # Signal that the detector is now idle.
                self.__detection_input_is_ready = False

    def __run_mapping(self) -> None:
        """Run the mapping thread."""
        frame_idx: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        skeleton_detector: Optional[RemoteSkeletonDetector] = RemoteSkeletonDetector() \
            if self.__detect_skeletons else None

        # Until termination is requested:
        while not self.__should_terminate.is_set():
            # If the server has a frame from the client that has not yet been processed:
            if self.__server.has_frames_now(self.__client_id):
                # Get the camera parameters from the server.
                height, width, _ = self.__server.get_image_shapes(self.__client_id)[0]
                intrinsics: Tuple[float, float, float, float] = self.__server.get_intrinsics(self.__client_id)[0]

                # Record them so that other threads have access to them.
                with self.__calibration_lock:
                    self.__image_size = (width, height)
                    self.__intrinsics = intrinsics

                # Pass the camera intrinsics to the depth estimator.
                self.__depth_estimator.set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                # Get the frame from the server.
                self.__server.get_frame(self.__client_id, receiver)
                colour_image: np.ndarray = receiver.get_rgb_image()
                mapping_w_t_c: np.ndarray = receiver.get_pose()

                # If we're using an ArUco+PnP relocaliser, try to use it to estimate the camera pose, so that the
                # map can later be aligned with the marker.
                if self.__aruco_relocaliser is not None:
                    aruco_from_camera: Optional[np.ndarray] = self.__aruco_relocaliser.estimate_pose(
                        colour_image, intrinsics
                    )
                    if aruco_from_camera is not None:
                        aruco_from_world_estimate: np.ndarray = aruco_from_camera @ np.linalg.inv(mapping_w_t_c)
                        self.__aruco_from_world_estimates.append(aruco_from_world_estimate)

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

                    # If we're debugging, show the depth image that is to be fused into the TSDF.
                    if self.__debug:
                        cv2.imshow("Estimated Depth Image", depopulated_depth_image / 5)
                        cv2.waitKey(1)

                    # Fuse the frame into the TSDF.
                    start = timer()

                    fx, fy, cx, cy = intrinsics
                    o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                        width, height, fx, fy, cx, cy
                    )
                    ReconstructionUtil.integrate_frame(
                        ImageUtil.flip_channels(colour_image), depopulated_depth_image,
                        np.linalg.inv(mapping_w_t_c), o3d_intrinsics, self.__tsdf
                    )

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
