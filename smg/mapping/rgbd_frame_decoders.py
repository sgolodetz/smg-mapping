import numpy as np

from typing import Optional

from smg.mapping import FrameMessage


class UncompressedRGBDFrameDecoder:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self):
        """TODO"""
        self.__rgb_image: Optional[np.ndarray] = None
        self.__depth_image: Optional[np.ndarray] = None

    # SPECIAL METHODS

    def __call__(self, frame_msg: FrameMessage) -> None:
        """
        TODO

        :param frame_msg:   TODO
        """
        self.__rgb_image = frame_msg.get_image_data(0).reshape(frame_msg.get_image_shapes()[0])
        self.__depth_image = frame_msg.get_image_data(1).view(np.uint16).reshape(frame_msg.get_image_shapes()[1][:2])

    # PUBLIC METHODS

    def get_depth_image(self) -> Optional[np.ndarray]:
        """
        TODO

        :return:    TODO
        """
        return self.__depth_image

    def get_rgb_image(self) -> Optional[np.ndarray]:
        """
        TODO

        :return:    TODO
        """
        return self.__rgb_image
