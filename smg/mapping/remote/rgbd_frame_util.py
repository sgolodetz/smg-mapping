import numpy as np
import struct

from typing import Tuple

from smg.utility import ImageUtil

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
        # TODO
        return msg

    @staticmethod
    def decompress_frame_message(msg: FrameMessage) -> FrameMessage:
        # TODO
        return msg

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
        msg.set_image_data(1, ImageUtil.to_short_depth(depth_image).reshape(-1).view(np.uint8))
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
