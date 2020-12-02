import numpy as np
import struct

from typing import Tuple

from smg.mapping import Message


class FrameHeaderMessage(Message):
    """A message containing the sizes (in bytes) and dimensions of the images for a single RGB-D frame."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a frame header message."""
        super().__init__()

        self.__image_byte_size_fmt: str = "<i"
        self.__image_size_fmt: str = "<ii"

        self.__depth_image_byte_size_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__image_byte_size_fmt)
        )
        self.__depth_image_size_segment: Tuple[int, int] = (
            Message._end_of(self.__depth_image_byte_size_segment), struct.calcsize(self.__image_size_fmt)
        )
        self.__rgb_image_byte_size_segment: Tuple[int, int] = (
            Message._end_of(self.__depth_image_size_segment), struct.calcsize(self.__image_byte_size_fmt)
        )
        self.__rgb_image_size_segment: Tuple[int, int] = (
            Message._end_of(self.__rgb_image_byte_size_segment), struct.calcsize(self.__image_size_fmt)
        )

        self._data = np.zeros(Message._end_of(self.__rgb_image_size_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def extract_depth_image_byte_size(self) -> int:
        """TODO"""
        return struct.unpack_from(
            self.__image_byte_size_fmt, self._data, self.__depth_image_byte_size_segment[0]
        )[0]

    def extract_depth_image_size(self) -> Tuple[int, int]:
        """TODO"""
        return struct.unpack_from(self.__image_size_fmt, self._data, self.__depth_image_size_segment[0])

    def extract_rgb_image_byte_size(self) -> int:
        """TODO"""
        return struct.unpack_from(
            self.__image_byte_size_fmt, self._data, self.__rgb_image_byte_size_segment[0]
        )[0]

    def extract_rgb_image_size(self) -> Tuple[int, int]:
        """TODO"""
        return struct.unpack_from(self.__image_size_fmt, self._data, self.__rgb_image_size_segment[0])

    def set_depth_image_byte_size(self, depth_image_byte_size: int) -> None:
        """
        Set the size (in bytes) of the depth image.

        :param depth_image_byte_size:   The size (in bytes) of the depth image.
        """
        struct.pack_into(
            self.__image_byte_size_fmt, self._data, self.__depth_image_byte_size_segment[0], depth_image_byte_size
        )

    def set_depth_image_size(self, depth_image_size: Tuple[int, int]) -> None:
        """
        Set the dimensions of the depth image.

        :param depth_image_size:    The dimensions of the depth image.
        """
        struct.pack_into(self.__image_size_fmt, self._data, self.__depth_image_size_segment[0], *depth_image_size)

    def set_rgb_image_byte_size(self, rgb_image_byte_size: int) -> None:
        """
        Set the size (in bytes) of the RGB image.

        :param rgb_image_byte_size: The size (in bytes of the RGB image).
        """
        struct.pack_into(
            self.__image_byte_size_fmt, self._data, self.__rgb_image_byte_size_segment[0], rgb_image_byte_size
        )

    def set_rgb_image_size(self, rgb_image_size: Tuple[int, int]) -> None:
        """
        Set the dimensions of the RGB image.

        :param rgb_image_size:  The dimensions of the RGB image.
        """
        struct.pack_into(self.__image_size_fmt, self._data, self.__rgb_image_size_segment[0], *rgb_image_size)
