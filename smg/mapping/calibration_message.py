import numpy as np
import struct

from itertools import chain
from typing import List, Tuple

from smg.mapping import Message


class CalibrationMessage(Message):
    """A message containing the calibration parameters for an RGB-D camera."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a calibration message."""
        super().__init__()

        self.__max_images = 2

        self.__image_byte_sizes_fmt: str = "<" + "i" * self.__max_images
        self.__image_shapes_fmt: str = "<" + "iii" * self.__max_images
        self.__intrinsics_fmt: str = "<" + "ffff" * self.__max_images

        self.__image_byte_sizes_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__image_byte_sizes_fmt)
        )
        self.__image_shapes_segment: Tuple[int, int] = (
            Message._end_of(self.__image_byte_sizes_segment), struct.calcsize(self.__image_shapes_fmt)
        )
        self.__intrinsics_segment: Tuple[int, int] = (
            Message._end_of(self.__image_shapes_segment), struct.calcsize(self.__intrinsics_fmt)
        )

        self._data = np.zeros(Message._end_of(self.__intrinsics_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def extract_image_byte_sizes(self) -> List[int]:
        """
        TODO

        :return:    TODO
        """
        return list(struct.unpack_from(self.__image_byte_sizes_fmt, self._data, self.__image_byte_sizes_segment[0]))

    def extract_image_shapes(self) -> List[Tuple[int, int, int]]:
        """
        Extract the image shapes from the message.

        :return:    The image shapes.
        """
        flat: List[int] = struct.unpack_from(
            self.__image_shapes_fmt, self._data, self.__image_shapes_segment[0]
        )
        return list(zip(flat[::3], flat[1::3], flat[2::3]))

    def extract_intrinsics(self) -> List[Tuple[float, float, float, float]]:
        """
        Extract the camera intrinsics from the message.

        :return:    The camera intrinsics, as (fx, fy, cx, cy) tuples.
        """
        flat: List[float] = struct.unpack_from(
            self.__intrinsics_fmt, self._data, self.__intrinsics_segment[0]
        )
        return list(zip(flat[::4], flat[1::4], flat[2::4], flat[3::4]))

    def get_max_images(self) -> int:
        """
        TODO

        :return:    TODO
        """
        return self.__max_images

    def set_image_byte_sizes(self, image_byte_sizes: List[int]) -> None:
        """
        TODO

        :param image_byte_sizes:    TODO
        """
        image_byte_sizes += [0] * (self.__max_images - len(image_byte_sizes))
        struct.pack_into(
            self.__image_byte_sizes_fmt, self._data, self.__image_byte_sizes_segment[0], *image_byte_sizes
        )

    def set_image_shapes(self, image_shapes: List[Tuple[int, int, int]]) -> None:
        """
        Copy the image shapes into the appropriate byte segment in the message.

        :param image_shapes:    The image shapes.
        """
        image_shapes += [(0, 0, 0)] * (self.__max_images - len(image_shapes))
        struct.pack_into(
            self.__image_shapes_fmt, self._data, self.__image_shapes_segment[0], *chain(*image_shapes)
        )

    def set_intrinsics(self, intrinsics: List[Tuple[float, float, float, float]]) -> None:
        """
        Copy the camera intrinsics into the appropriate byte segment in the message.

        :param intrinsics:  The camera intrinsics.
        """
        intrinsics += [(0, 0, 0, 0)] * (self.__max_images - len(intrinsics))
        struct.pack_into(
            self.__intrinsics_fmt, self._data, self.__intrinsics_segment[0], *chain(*intrinsics)
        )
