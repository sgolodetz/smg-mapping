import cv2
import detectron2
import numpy as np
import open3d as o3d
import threading
import time

from detectron2.structures import Instances
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.base import RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.comms.skeletons import RemoteSkeletonDetector
from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
from smg.mvdepthnet import MonocularDepthEstimator
from smg.open3d import ReconstructionUtil
from smg.skeletons import Skeleton, SkeletonUtil
from smg.utility import GeometryUtil, ImageUtil, RGBDSequenceUtil


class MVDepthOpen3DMappingSystem:
    """A mapping system that estimates depths using MVDepthNet and reconstructs an Open3D TSDF."""

    # CONSTRUCTOR

    def __init__(self, server: MappingServer, depth_estimator: MonocularDepthEstimator, *,
                 detect_objects: bool = False, output_dir: Optional[str] = None, save_frames: bool = False):
        """
        Construct a mapping system that estimates depths using MVDepthNet and reconstructs an Open3D TSDF.

        :param server:          The mapping server.
        :param depth_estimator: The monocular depth estimator.
        :param detect_objects:  Whether to detect 3D objects.
        :param output_dir:      An optional directory into which to save output files.
        :param save_frames:     Whether to save the sequence of frames used to reconstruct the TSDF.
        """
        self.__client_id: int = 0
        self.__depth_estimator: MonocularDepthEstimator = depth_estimator
        self.__detect_objects: bool = detect_objects
        self.__output_dir: Optional[str] = output_dir
        self.__save_frames: bool = save_frames
        self.__server: MappingServer = server
        self.__should_terminate: threading.Event = threading.Event()
        self.__skeleton_detector: RemoteSkeletonDetector = RemoteSkeletonDetector()

        # The image size and camera intrinsics, together with their lock.
        self.__image_size: Optional[Tuple[int, int]] = None
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__parameters_lock: threading.Lock = threading.Lock()

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
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        self.__scene_lock: threading.Lock = threading.Lock()

        # The threads and conditions.
        self.__detection_thread: Optional[threading.Thread] = None
        self.__detection_input_is_ready: bool = False
        self.__detection_input_ready: threading.Condition = threading.Condition(self.__detection_lock)

        self.__mapping_thread: Optional[threading.Thread] = None

    # PUBLIC METHODS

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
            # If the server has any frames from the client that have not yet been processed, get the colour image
            # from the most recent one.
            if self.__server.peek_newest_frame(self.__client_id, receiver):
                newest_colour_image = receiver.get_rgb_image()

            # If we've ever seen a frame:
            if newest_colour_image is not None:
                # Show the most recent colour image.
                cv2.imshow("MVDepth -> Open3D Server", newest_colour_image)
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
                    # Share the instance segmentation and the detected 3D objects with other threads.
                    self.__instance_segmentation = instance_segmentation.copy()
                    self.__objects = objects

                # Signal that the detector is now idle.
                self.__detection_input_is_ready = False

    def __run_mapping(self) -> None:
        """Run the mapping thread."""
        frame_idx: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

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

                # Set the camera calibration for the skeleton detector.
                # FIXME: Do this once.
                self.__skeleton_detector.set_calibration((width, height), intrinsics)

                # Start trying to detect any skeletons in the colour image. If this fails, skip this frame.
                if not self.__skeleton_detector.begin_detection(colour_image, mapping_w_t_c):
                    time.sleep(0.01)
                    continue

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

                    # Remove any detected people.
                    skeletons, people_mask = self.__skeleton_detector.end_detection()
                    depopulated_depth_image: np.ndarray = estimated_depth_image.copy()
                    if people_mask is not None:
                        depopulated_depth_image = SkeletonUtil.depopulate_depth_image(
                            estimated_depth_image, people_mask
                        )

                    # Fuse the frame into the TSDF.
                    start = timer()

                    fx, fy, cx, cy = intrinsics
                    o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                        width, height, fx, fy, cx, cy
                    )
                    ReconstructionUtil.integrate_frame(
                        ImageUtil.flip_channels(colour_image), depopulated_depth_image, np.linalg.inv(mapping_w_t_c),
                        o3d_intrinsics, self.__tsdf
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
