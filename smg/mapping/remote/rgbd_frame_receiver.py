import numpy as np

from typing import Optional

from .frame_message import FrameMessage
from .rgbd_frame_message_util import RGBDFrameMessageUtil


class RGBDFrameReceiver:
    """A receiver of RGB-D frames, used to extract them from frame messages and store them for later use."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct an RGB-D frame receiver."""
        self.__rgb_image: Optional[np.ndarray] = None
        self.__depth_image: Optional[np.ndarray] = None
        self.__pose: Optional[np.ndarray] = None

    # SPECIAL METHODS

    def __call__(self, msg: FrameMessage) -> None:
        """
        Apply the receiver to a frame message.

        .. note::
            This extracts an RGB-D frame from the frame message and stores it in the receiver.

        :param msg: The frame message.
        """
        self.__frame_idx, self.__rgb_image, self.__depth_image, self.__pose = \
            RGBDFrameMessageUtil.extract_frame_data(msg)

    # PUBLIC METHODS

    def get_depth_image(self) -> np.ndarray:
        """
        Get the depth image of the RGB-D frame.

        :return:    The depth image of the RGB-D frame.
        """
        return self.__depth_image

    def get_pose(self) -> np.ndarray:
        """
        Get the pose of the RGB-D frame.

        :return:    The pose of the RGB-D frame.
        """
        return self.__pose

    def get_rgb_image(self) -> np.ndarray:
        """
        Get the colour image of the RGB-D frame.

        :return:    The colour image of the RGB-D frame.
        """
        return self.__rgb_image
