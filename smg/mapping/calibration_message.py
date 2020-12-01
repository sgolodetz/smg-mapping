import numpy as np
import struct

from typing import Tuple

from smg.mapping import Message


class CalibrationMessage(Message):
    """A message containing the calibration parameters for an RGB-D camera."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a calibration message."""
        self.__intrinsics_fmt: str = "<ffff"
        self.__intrinsics_segment: Tuple[int, int] = (0, struct.calcsize(self.__intrinsics_fmt))
        self.__data: np.ndarray = np.zeros(Message._end_of(self.__intrinsics_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def extract_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Extract the camera intrinsics from the message.

        :return:    The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        return struct.unpack_from(self.__intrinsics_fmt, self.__data, self.__intrinsics_segment[0])

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

    def set_intrinsics(self, intrinsics: Tuple[float, float, float, float]) -> None:
        """
        Copy the camera intrinsics into the appropriate byte segment in the message.

        :param intrinsics:  The camera intrinsics.
        """
        struct.pack_into(self.__intrinsics_fmt, self.__data, self.__intrinsics_segment[0], *intrinsics)
