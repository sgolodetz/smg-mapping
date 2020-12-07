import numpy as np
import struct

from itertools import chain
from typing import List, Tuple

from .message import Message


class CalibrationMessage(Message):
    """A message containing the calibration parameters for the images that we're using."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a calibration message."""
        super().__init__()

        # Set the maximum number of images that can be transmitted in a frame message. We're currently only
        # sending RGB-D images, which is why this is set to 2, but it can be increased if needed.
        self.__max_images = 2

        # The image shapes segment consists of a list of tuples [(h_1,w_1,ch_1), ...].
        self.__image_shapes_fmt: str = "<" + "iii" * self.__max_images

        # The intrinsics segment consists of a list of tuples [(fx_1,fy_1,cx_1,cy_1), ...].
        self.__intrinsics_fmt: str = "<" + "ffff" * self.__max_images

        # The element byte sizes segment consists of a list of integers [bs_1,...], in which bs_i
        # denotes the size of an individual element in image i, such that the overall byte size of
        # image i is h_i * w_i * ch_i * bs_i. Crucially, this is not the size of an entire pixel,
        # which would be ch_i * bs_i.
        self.__element_byte_sizes_fmt: str = "<" + "i" * self.__max_images

        self.__image_shapes_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__image_shapes_fmt)
        )
        self.__intrinsics_segment: Tuple[int, int] = (
            Message._end_of(self.__image_shapes_segment), struct.calcsize(self.__intrinsics_fmt)
        )
        self.__element_byte_sizes_segment: Tuple[int, int] = (
            Message._end_of(self.__intrinsics_segment), struct.calcsize(self.__element_byte_sizes_fmt)
        )

        self._data = np.zeros(Message._end_of(self.__element_byte_sizes_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def get_image_shapes(self) -> List[Tuple[int, int, int]]:
        """
        Get the image shapes from the message.

        :return:    The image shapes.
        """
        flat: List[int] = struct.unpack_from(
            self.__image_shapes_fmt, self._data, self.__image_shapes_segment[0]
        )
        return list(zip(flat[::3], flat[1::3], flat[2::3]))

    def get_intrinsics(self) -> List[Tuple[float, float, float, float]]:
        """
        Get the camera intrinsics from the message.

        :return:    The camera intrinsics, as (fx, fy, cx, cy) tuples.
        """
        flat: List[float] = struct.unpack_from(
            self.__intrinsics_fmt, self._data, self.__intrinsics_segment[0]
        )
        return list(zip(flat[::4], flat[1::4], flat[2::4], flat[3::4]))

    def get_max_images(self) -> int:
        """
        Get the maximum number of images that can be transmitted in a frame message.

        :return:    The maximum number of images that can be transmitted in a frame message.
        """
        return self.__max_images

    def get_uncompressed_image_byte_sizes(self) -> List[int]:
        """
        Get the (uncompressed) byte sizes of the images.

        .. note::
            These are calculated from the image shapes and element byte sizes.

        :return:    The (unocmpressed) byte sizes of the images.
        """
        image_shapes: List[Tuple[int, int, int]] = self.get_image_shapes()
        element_byte_sizes: List[int] = list(
            struct.unpack_from(self.__element_byte_sizes_fmt, self._data, self.__element_byte_sizes_segment[0])
        )
        return [np.prod(s) * b for s, b in zip(image_shapes, element_byte_sizes)]

    def set_element_byte_sizes(self, element_byte_sizes: List[int]) -> None:
        """
        Copy the element byte sizes for the different images into the appropriate byte segment in the message.

        .. note::
            The element byte size bs of an image is such that the overall byte size of an image with
            shape (h,w,ch) is h * w * ch * bs. Note that it's not the same as the pixel size, which
            would be ch * bs.

        :param element_byte_sizes:  The element byte sizes for the different images.
        """
        element_byte_sizes += [0] * (self.__max_images - len(element_byte_sizes))
        struct.pack_into(
            self.__element_byte_sizes_fmt, self._data, self.__element_byte_sizes_segment[0], *element_byte_sizes
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
