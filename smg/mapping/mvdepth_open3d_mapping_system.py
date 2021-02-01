import cv2
import detectron2
import numpy as np
import open3d as o3d
import threading

from timeit import default_timer as timer
from typing import List, Optional, Tuple

from detectron2.structures import Instances

from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
from smg.mapping.remote import MappingServer, RGBDFrameReceiver
from smg.mvdepthnet import MonocularDepthEstimator
from smg.open3d import ReconstructionUtil
from smg.utility import GeometryUtil, ImageUtil, RGBDSequenceUtil


class MVDepthOpen3DMappingSystem:
    """A mapping system that estimates depths using MVDepthNet and reconstructs an Open3D TSDF."""

    # CONSTRUCTOR

    def __init__(self, *, depth_estimator: MonocularDepthEstimator, output_dir: Optional[str] = None,
                 server: MappingServer):
        """
        TODO

        :param depth_estimator: TODO
        :param output_dir:      TODO
        :param server:          TODO
        """
        self.__depth_estimator: MonocularDepthEstimator = depth_estimator
        self.__output_dir: Optional[str] = output_dir
        self.__server: MappingServer = server
        self.__should_terminate: bool = False

        self.__tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        self.__objects: List[ObjectDetector3D.Object3D] = []

        self.__detection_lock: threading.Lock = threading.Lock()
        self.__detection_colour_image: Optional[np.ndarray] = None
        self.__detection_depth_image: Optional[np.ndarray] = None
        self.__detection_pose: Optional[np.ndarray] = None
        self.__detection_intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__detection_input_ready: threading.Condition = threading.Condition(self.__detection_lock)
        self.__detection_output_image: Optional[np.ndarray] = None
        self.__detection_required: bool = False

        self.__detection_thread: Optional[threading.Thread] = None

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the mapping system's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the mapping system at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> Tuple[o3d.pipelines.integration.ScalableTSDFVolume, List[ObjectDetector3D.Object3D]]:
        client_id: int = 0
        colour_image: Optional[np.ndarray] = None
        frame_idx: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

        # Start the detection thread.
        self.__detection_thread = threading.Thread(target=self.__run_detection)
        self.__detection_thread.start()

        # Until the mapping system should terminate:
        while not self.__should_terminate:
            # If the server has a frame from the client that has not yet been processed:
            if self.__server.has_frames_now(client_id):
                # Get the camera intrinsics from the server, and pass them to the depth estimator.
                intrinsics: Tuple[float, float, float, float] = self.__server.get_intrinsics(client_id)[0]
                self.__depth_estimator.set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                # Get the frame from the server.
                self.__server.get_frame(client_id, receiver)
                colour_image = receiver.get_rgb_image()
                tracker_w_t_c: np.ndarray = receiver.get_pose()

                # If an output directory has been specified, save the frame to disk.
                if self.__output_dir is not None:
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

                    height, width = estimated_depth_image.shape
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
                            if self.__detection_output_image is not None:
                                cv2.imshow("Detection Output Image", self.__detection_output_image)
                            self.__detection_colour_image = colour_image.copy()
                            self.__detection_depth_image = estimated_depth_image.copy()
                            self.__detection_pose = tracker_w_t_c.copy()
                            self.__detection_intrinsics = intrinsics
                            self.__detection_required = True
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
                while not self.__detection_required:
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
                self.__detection_output_image = instance_segmenter.draw_raw_instances(
                    raw_instances, self.__detection_colour_image
                )

                start = timer()

                # TODO: Comment here.
                instances: List[InstanceSegmenter.Instance] = instance_segmenter.parse_raw_instances(raw_instances)
                instances = [instance for instance in instances if instance.label != "book"]
                self.__objects += object_detector.lift_to_3d(
                    instances, self.__detection_depth_image, self.__detection_pose, self.__detection_intrinsics
                )

                end = timer()
                print(f"  - Lifting Time: {end - start}s")

                self.__detection_required = False
