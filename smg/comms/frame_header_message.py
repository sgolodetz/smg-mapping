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

        # The image byte sizes segment consists of a list of integers [bs_1,...], in which bs_i denotes the
        # overall byte size of image i as stored in the frame message (e.g. potentially after compression).
        self.__image_byte_sizes_fmt: str = "<" + "i" * max_images

        # The image shapes segment consists of a list of tuples [(h_1,w_1,ch_1), ...].
        self.__image_shapes_fmt: str = "<" + "iii" * max_images

        self.__image_byte_sizes_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__image_byte_sizes_fmt)
        )
        self.__image_shapes_segment: Tuple[int, int] = (
            Message._end_of(self.__image_byte_sizes_segment), struct.calcsize(self.__image_shapes_fmt)
        )

        self._data = np.zeros(Message._end_of(self.__image_shapes_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def get_image_byte_sizes(self) -> List[int]:
        """
        Get the byte sizes of the images (as stored in the frame message).

        :return:    The byte sizes of the images (as stored in the frame message).
        """
        return list(struct.unpack_from(self.__image_byte_sizes_fmt, self._data, self.__image_byte_sizes_segment[0]))

    def get_image_shapes(self) -> List[Tuple[int, int, int]]:
        """
        Get the image shapes from the message.

        :return:    The image shapes.
        """
        flat: List[int] = struct.unpack_from(
            self.__image_shapes_fmt, self._data, self.__image_shapes_segment[0]
        )
        return list(zip(flat[::3], flat[1::3], flat[2::3]))

    def set_image_byte_sizes(self, image_byte_sizes: List[int]) -> None:
        """
        Copy the byte sizes of the images into the appropriate byte segment in the message.

        :param image_byte_sizes:    The byte sizes of the images (as stored in the frame message).
        """
        struct.pack_into(
            self.__image_byte_sizes_fmt, self._data, self.__image_byte_sizes_segment[0], *image_byte_sizes
        )

    def set_image_shapes(self, image_shapes: List[Tuple[int, int, int]]) -> None:
        """
        Copy the image shapes into the appropriate byte segment in the message.

        :param image_shapes:    The image shapes.
        """
        struct.pack_into(
            self.__image_shapes_fmt, self._data, self.__image_shapes_segment[0], *chain(*image_shapes)
        )
