import numpy as np
import struct

from typing import Tuple

from smg.mapping import Message


class CalibrationMessage(Message):
    """A message containing the calibration parameters for an RGB-D camera."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a calibration message."""
        super().__init__()

        self.__image_size_fmt: str = "<ii"
        self.__intrinsics_fmt: str = "<ffff"

        self.__image_size_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__image_size_fmt)
        )
        self.__intrinsics_segment: Tuple[int, int] = (
            Message._end_of(self.__image_size_segment), struct.calcsize(self.__intrinsics_fmt)
        )

        self._data = np.zeros(Message._end_of(self.__intrinsics_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def extract_image_size(self) -> Tuple[int, int]:
        """
        Extract the image size from the message.

        :return:    The image size.
        """
        return struct.unpack_from(self.__image_size_fmt, self._data, self.__image_size_segment[0])

    def extract_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Extract the camera intrinsics from the message.

        :return:    The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        return struct.unpack_from(self.__intrinsics_fmt, self._data, self.__intrinsics_segment[0])

    def set_image_size(self, image_size: Tuple[int, int]) -> None:
        """
        Copy the image size into the appropriate byte segment in the message.

        :param image_size:  The image size.
        """
        struct.pack_into(self.__image_size_fmt, self._data, self.__image_size_segment[0], *image_size)

    def set_intrinsics(self, intrinsics: Tuple[float, float, float, float]) -> None:
        """
        Copy the camera intrinsics into the appropriate byte segment in the message.

        :param intrinsics:  The camera intrinsics.
        """
        struct.pack_into(self.__intrinsics_fmt, self._data, self.__intrinsics_segment[0], *intrinsics)
