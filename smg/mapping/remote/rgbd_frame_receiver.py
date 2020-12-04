import numpy as np

from typing import Optional

from .frame_message import FrameMessage
from .rgbd_frame_util import RGBDFrameUtil


class RGBDFrameReceiver:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self):
        """TODO"""
        self.__rgb_image: Optional[np.ndarray] = None
        self.__depth_image: Optional[np.ndarray] = None
        self.__pose: Optional[np.ndarray] = None

    # SPECIAL METHODS

    def __call__(self, msg: FrameMessage) -> None:
        """
        TODO

        :param msg: TODO
        """
        self.__frame_idx, self.__rgb_image, self.__depth_image, self.__pose = RGBDFrameUtil.extract_frame_data(msg)

    # PUBLIC METHODS

    def get_depth_image(self) -> np.ndarray:
        """
        TODO

        :return:    TODO
        """
        return self.__depth_image

    def get_pose(self) -> np.ndarray:
        """
        TODO

        :return:    TODO
        """
        return self.__pose

    def get_rgb_image(self) -> np.ndarray:
        """
        TODO

        :return:    TODO
        """
        return self.__rgb_image
