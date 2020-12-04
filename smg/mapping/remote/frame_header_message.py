import numpy as np
import struct

from itertools import chain
from typing import List, Tuple

from .message import Message


class FrameHeaderMessage(Message):
    """A message containing the sizes (in bytes) and shapes of the images for a single frame."""

    # CONSTRUCTOR

    def __init__(self, max_images: int):
        """Construct a frame header message."""
        super().__init__()

        self.__image_byte_sizes_fmt: str = "<" + "i" * max_images
        self.__image_shapes_fmt: str = "<" + "iii" * max_images

        self.__image_byte_sizes_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__image_byte_sizes_fmt)
        )
        self.__image_shapes_segment: Tuple[int, int] = (
            Message._end_of(self.__image_byte_sizes_segment), struct.calcsize(self.__image_shapes_fmt)
        )

        self._data = np.zeros(Message._end_of(self.__image_shapes_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def extract_image_byte_sizes(self) -> List[int]:
        """
        TODO

        :return:    TODO
        """
        return list(struct.unpack_from(self.__image_byte_sizes_fmt, self._data, self.__image_byte_sizes_segment[0]))

    def extract_image_shapes(self) -> List[Tuple[int, int, int]]:
        """
        TODO

        :return:    TODO
        """
        flat: List[int] = struct.unpack_from(
            self.__image_shapes_fmt, self._data, self.__image_shapes_segment[0]
        )
        return list(zip(flat[::3], flat[1::3], flat[2::3]))

    def set_image_byte_sizes(self, image_byte_sizes: List[int]) -> None:
        """
        TODO

        :param image_byte_sizes:    TODO
        """
        struct.pack_into(
            self.__image_byte_sizes_fmt, self._data, self.__image_byte_sizes_segment[0], *image_byte_sizes
        )

    def set_image_shapes(self, image_shapes: List[Tuple[int, int, int]]) -> None:
        """
        TODO

        :param image_shapes:    TODO
        """
        struct.pack_into(
            self.__image_shapes_fmt, self._data, self.__image_shapes_segment[0], *chain(*image_shapes)
        )
