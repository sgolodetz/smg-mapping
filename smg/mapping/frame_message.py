import numpy as np

from typing import Tuple

from smg.mapping import Message


class FrameMessage(Message):
    """A message containing a single frame of RGB-D data (frame index + pose + RGB-D)."""

    # CONSTRUCTOR

    def __init__(self):
        self.__frame_index_fmt: str = "<i"
        self.__pose_fmt: str = "<ffffffffffff"
        # TODO
        pass

    # PUBLIC METHODS

    def get_data(self) -> np.ndarray:
        """
        Get the message data.

        :return:    Get the message data.
        """
        # TODO
        pass

    def get_size(self) -> int:
        """
        Get the size of the message.

        :return:    The size of the message.
        """
        # TODO
        pass
