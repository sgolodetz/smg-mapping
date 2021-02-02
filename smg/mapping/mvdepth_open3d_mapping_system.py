import cv2
import detectron2
import numpy as np
import open3d as o3d
import threading

from detectron2.structures import Instances
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
from smg.mapping.remote import MappingServer, RGBDFrameReceiver
from smg.mvdepthnet import MonocularDepthEstimator
from smg.open3d import ReconstructionUtil
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
        self.__depth_estimator: MonocularDepthEstimator = depth_estimator
        self.__detect_objects: bool = detect_objects
        self.__objects: List[ObjectDetector3D.Object3D] = []
        self.__output_dir: Optional[str] = output_dir
        self.__save_frames: bool = save_frames
        self.__server: MappingServer = server
        self.__should_terminate: bool = False
        self.__tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # The image size and camera intrinsics, together with their lock.
        self.__image_size: Optional[Tuple[int, int]] = None
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__parameters_lock: threading.Lock = threading.Lock()

        # The detection inputs/outputs, and their lock.
        self.__detection_colour_image: Optional[np.ndarray] = None
        self.__detection_depth_image: Optional[np.ndarray] = None
        self.__detection_w_t_c: Optional[np.ndarray] = None
        self.__instance_segmentation: Optional[np.ndarray] = None
        self.__detection_lock: threading.Lock = threading.Lock()

        # The threads and conditions.
        self.__detection_thread: Optional[threading.Thread] = None
        self.__detection_input_is_ready: bool = False
        self.__detection_input_ready: threading.Condition = threading.Condition(self.__detection_lock)

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the mapping system's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the mapping system at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> Tuple[o3d.pipelines.integration.ScalableTSDFVolume, List[ObjectDetector3D.Object3D]]:
        """
        Run the mapping system.

        :return:    The results of the reconstruction process, as a (TSDF, list of 3D objects) tuple.
        """
        client_id: int = 0
        colour_image: Optional[np.ndarray] = None
        frame_idx: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

        # If we're detecting 3D objects, start the detection thread.
        if self.__detect_objects:
            self.__detection_thread = threading.Thread(target=self.__run_detection)
            self.__detection_thread.start()

        # Until the mapping system should terminate:
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
                colour_image = receiver.get_rgb_image()
                tracker_w_t_c: np.ndarray = receiver.get_pose()

                # If an output directory was specified and we're saving frames, save the frame to disk.
                if self.__output_dir is not None and self.__save_frames:
                    depth_image: np.ndarray = receiver.get_depth_image()
                    RGBDSequenceUtil.save_frame(
                        frame_idx, self.__output_dir, colour_image, depth_image, tracker_w_t_c,
                        colour_intrinsics=intrinsics, depth_intrinsics=intrinsics
                    )
                    frame_idx += 1

                # Try to estimate a depth image for the frame.
                estimated_depth_image: Optional[np.ndarray] = self.__depth_estimator.estimate_depth(
                    colour_image, tracker_w_t_c
                )

                # If a depth image was successfully estimated:
                if estimated_depth_image is not None:
                    # Limit its range to 3m (more distant points can be unreliable).
                    estimated_depth_image = np.where(estimated_depth_image <= 3.0, estimated_depth_image, 0.0)

                    # Fuse the frame into the TSDF.
                    start = timer()

                    fx, fy, cx, cy = intrinsics
                    o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
                        width, height, fx, fy, cx, cy
                    )
                    ReconstructionUtil.integrate_frame(
                        ImageUtil.flip_channels(colour_image), estimated_depth_image, np.linalg.inv(tracker_w_t_c),
                        o3d_intrinsics, self.__tsdf
                    )

                    end = timer()
                    print(f"  - Fusion Time: {end - start}s")

                    # TODO: Comment here.
                    acquired: bool = self.__detection_lock.acquire(blocking=False)
                    if acquired:
                        try:
                            if self.__instance_segmentation is not None:
                                cv2.imshow("Instance Segmentation", self.__instance_segmentation)
                            self.__detection_colour_image = colour_image.copy()
                            self.__detection_depth_image = estimated_depth_image.copy()
                            self.__detection_w_t_c = tracker_w_t_c.copy()
                            self.__detection_input_is_ready = True
                            self.__detection_input_ready.notify()
                        finally:
                            self.__detection_lock.release()

            # TODO: Comment here.
            if colour_image is not None:
                # TODO: Comment here.
                cv2.imshow("MVDepth -> Open3D Mapping System", colour_image)
                c: int = cv2.waitKey(1)

                # TODO: Comment here.
                if c == ord('v'):
                    cv2.destroyAllWindows()
                    return self.terminate()

    def terminate(self) -> Tuple[o3d.pipelines.integration.ScalableTSDFVolume, List[ObjectDetector3D.Object3D]]:
        """
        Destroy the mapping system.

        :return:    TODO
        """
        if not self.__should_terminate:
            # TODO: Comment here.
            self.__should_terminate = True

            # TODO: Comment here.
            if self.__detection_thread is not None:
                self.__detection_thread.join()

        return self.__tsdf, self.__objects

    # PRIVATE METHODS

    def __run_detection(self) -> None:
        print("Starting object detector...", flush=True)
        instance_segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()
        object_detector: ObjectDetector3D = ObjectDetector3D(instance_segmenter)
        print("Object detector started", flush=True)

        while not self.__should_terminate:
            with self.__detection_lock:
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

                # Draw the 2D instances so that they can be shown to the user.
                self.__instance_segmentation = instance_segmenter.draw_raw_instances(
                    raw_instances, self.__detection_colour_image
                )

                # Get the camera intrinsics.
                with self.__parameters_lock:
                    intrinsics: Tuple[float, float, float, float] = self.__intrinsics

                start = timer()

                # TODO: Comment here.
                instances: List[InstanceSegmenter.Instance] = instance_segmenter.parse_raw_instances(raw_instances)
                instances = [instance for instance in instances if instance.label != "book"]
                self.__objects += object_detector.lift_to_3d(
                    instances, self.__detection_depth_image, self.__detection_w_t_c, intrinsics
                )

                end = timer()
                print(f"  - Lifting Time: {end - start}s")

                self.__detection_input_is_ready = False
