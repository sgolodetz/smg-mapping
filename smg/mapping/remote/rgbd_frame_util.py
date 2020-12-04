import cv2
import numpy as np
import struct

from typing import Tuple

from .calibration_message import CalibrationMessage
from .frame_message import FrameMessage


class RGBDFrameUtil:
    """TODO"""

    # PUBLIC STATIC METHODS

    @staticmethod
    def compress_frame_message(msg: FrameMessage) -> FrameMessage:
        """
        TODO

        :param msg: TODO
        :return:    TODO
        """
        # TODO: Comment here.
        frame_idx, rgb_image, depth_image, pose = RGBDFrameUtil.extract_frame_data(msg)

        # TODO: Comment here.
        compressed_rgb_image: np.ndarray = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        compressed_depth_image: np.ndarray = cv2.imencode(".png", depth_image)[1]

        # TODO: Comment here.
        compressed_msg: FrameMessage = FrameMessage(
            msg.get_image_shapes(), [len(compressed_rgb_image), len(compressed_depth_image)]
        )
        compressed_msg.set_frame_index(frame_idx)
        compressed_msg.set_image_data(0, compressed_rgb_image.flatten())
        compressed_msg.set_pose(0, pose)
        compressed_msg.set_image_data(1, compressed_depth_image.flatten())
        compressed_msg.set_pose(1, pose)

        return compressed_msg

    @staticmethod
    def decompress_frame_message(msg: FrameMessage) -> FrameMessage:
        # TODO
        frame_idx: int = msg.get_frame_index()
        compressed_rgb_image: np.ndarray = msg.get_image_data(0)
        compressed_depth_image: np.ndarray = msg.get_image_data(1)
        pose: np.ndarray = msg.get_pose(0)

        # TODO
        rgb_image: np.ndarray = cv2.imdecode(compressed_rgb_image, cv2.IMREAD_COLOR)
        depth_image: np.ndarray = cv2.imdecode(compressed_depth_image, cv2.IMREAD_ANYDEPTH).astype(np.uint16)

        # TODO
        decompressed_msg: FrameMessage = FrameMessage(
            msg.get_image_shapes(),
            [rgb_image.nbytes, depth_image.nbytes]
        )
        RGBDFrameUtil.fill_frame_message(frame_idx, rgb_image, depth_image, pose, decompressed_msg)

        return decompressed_msg

    @staticmethod
    def extract_frame_data(msg: FrameMessage) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO

        :param msg: TODO
        :return:    TODO
        """
        frame_idx: int = msg.get_frame_index()
        rgb_image: np.ndarray = msg.get_image_data(0).reshape(msg.get_image_shapes()[0])
        depth_image: np.ndarray = msg.get_image_data(1).view(np.uint16).reshape(msg.get_image_shapes()[1][:2])
        pose: np.ndarray = msg.get_pose(0)
        return frame_idx, rgb_image, depth_image, pose

    @staticmethod
    def fill_frame_message(frame_idx: int, rgb_image: np.ndarray, depth_image: np.ndarray, pose: np.ndarray,
                           msg: FrameMessage) -> None:
        """
        TODO

        :param frame_idx:   TODO
        :param rgb_image:   TODO
        :param depth_image: TODO
        :param pose:        TODO
        :param msg:         TODO
        """
        msg.set_frame_index(frame_idx)
        msg.set_image_data(0, rgb_image.reshape(-1))
        msg.set_pose(0, pose)
        msg.set_image_data(1, depth_image.reshape(-1).view(np.uint8))
        msg.set_pose(1, pose)

    @staticmethod
    def make_calibration_message(rgb_image_size: Tuple[int, int], depth_image_size: Tuple[int, int],
                                 rgb_intrinsics: Tuple[float, float, float, float],
                                 depth_intrinsics: Tuple[float, float, float, float]) -> CalibrationMessage:
        """
        TODO

        :param rgb_image_size:      TODO
        :param depth_image_size:    TODO
        :param rgb_intrinsics:      TODO
        :param depth_intrinsics:    TODO
        :return:                    TODO
        """
        calib_msg: CalibrationMessage = CalibrationMessage()

        # noinspection PyTypeChecker
        calib_msg.set_image_shapes([rgb_image_size[::-1] + (3,), depth_image_size[::-1] + (1,)])
        calib_msg.set_intrinsics([rgb_intrinsics, depth_intrinsics])
        calib_msg.set_pixel_byte_sizes([struct.calcsize("<B"), struct.calcsize("<H")])

        return calib_msg
