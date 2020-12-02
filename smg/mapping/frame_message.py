import numpy as np
import struct

from typing import Optional, Tuple

from smg.mapping import Message


class FrameMessage(Message):
    """A message containing a single frame of RGB-D data (frame index + pose + RGB-D)."""

    # CONSTRUCTOR

    def __init__(self, rgb_image_size: Tuple[int, int], depth_image_size: Tuple[int, int], *,
                 rgb_image_byte_size: Optional[int] = None, depth_image_byte_size: Optional[int] = None):
        """
        Construct an RGB-D frame message.

        .. note::
            The images may be compressed, so we allow the byte sizes of the images to be passed in independently
            of their dimensions. If separate byte sizes are not specified, it will be assumed that the images are
            not compressed, and the bytes sizes will be inferred from the images' dimensions in the obvious way.

        :param rgb_image_size:          The dimensions of the frame's RGB image, as a (width, height) tuple.
        :param depth_image_size:        The dimensions of the frame's depth image, as a (width, height) tuple.
        :param rgb_image_byte_size:     The size (in bytes) of the memory used to store the frame's RGB image.
        :param depth_image_byte_size:   The size (in bytes) of the memory used to store the frame's depth image.
        """
        if rgb_image_byte_size is None:
            rgb_image_byte_size = rgb_image_size[0] * rgb_image_size[1] * struct.calcsize("<BBB")
        if depth_image_byte_size is None:
            depth_image_byte_size = depth_image_size[0] * depth_image_size[1] * struct.calcsize("<f")

        self.__frame_index_fmt: str = "<i"
        self.__pose_fmt: str = "<ffffffffffff"

        self.__frame_index_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__frame_index_fmt)
        )
        self.__pose_segment: Tuple[int, int] = (
            Message._end_of(self.__frame_index_segment), struct.calcsize(self.__pose_fmt)
        )
        self.__rgb_image_segment: Tuple[int, int] = (
            Message._end_of(self.__pose_segment), rgb_image_byte_size
        )
        self.__depth_image_segment: Tuple[int, int] = (
            Message._end_of(self.__rgb_image_segment), depth_image_byte_size
        )

        self.__data: np.ndarray = np.zeros(Message._end_of(self.__depth_image_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def get_data(self) -> np.ndarray:
        """
        Get the message data.

        :return:    Get the message data.
        """
        return self.__data

    def get_size(self) -> int:
        """
        Get the size of the message.

        :return:    The size of the message.
        """
        return len(self.__data)
