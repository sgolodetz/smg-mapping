import numpy as np
import struct

from typing import List, Tuple

from smg.mapping import Message


class FrameHeaderMessage(Message):
    """A message containing the sizes (in bytes) and dimensions of the images for a single frame."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a frame header message."""
        super().__init__()

        max_images: int = 2
        self.__image_byte_size_fmt: str = "<i"
        self.__image_byte_sizes_fmt: str = "<" + "i" * max_images
        self.__image_size_fmt: str = "<iii"
        self.__image_sizes_fmt: str = "<" + "iii" * max_images

        self.__image_byte_sizes_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__image_byte_sizes_fmt)
        )
        self.__image_sizes_segment: Tuple[int, int] = (
            Message._end_of(self.__image_byte_sizes_segment), struct.calcsize(self.__image_sizes_fmt)
        )

        self._data = np.zeros(Message._end_of(self.__image_sizes_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def extract_image_byte_sizes(self) -> List[int]:
        """
        TODO

        :return:    TODO
        """
        return list(struct.unpack_from(self.__image_byte_sizes_fmt, self._data, self.__image_byte_sizes_segment[0]))

    def extract_image_sizes(self) -> List[Tuple[int, int, int]]:
        """
        TODO

        :return:    TODO
        """
        flat: List[int] = struct.unpack_from(
            self.__image_sizes_fmt, self._data, self.__image_sizes_segment[0]
        )
        return list(zip(flat[::3], flat[1::3], flat[2::3]))

    def set_image_byte_size(self, image_idx: int, image_byte_size: int) -> None:
        """
        TODO

        :param image_idx:       TODO
        :param image_byte_size: TODO
        """
        struct.pack_into(
            self.__image_byte_size_fmt, self._data,
            self.__image_byte_sizes_segment[0] + image_idx * struct.calcsize(self.__image_byte_size_fmt),
            image_byte_size
        )

    def set_image_size(self, image_idx: int, image_size: Tuple[int, int, int]) -> None:
        """
        TODO

        :param image_idx:   TODO
        :param image_size:  TODO
        """
        struct.pack_into(
            self.__image_size_fmt, self._data,
            self.__image_sizes_segment[0] + image_idx * struct.calcsize(self.__image_size_fmt),
            *image_size
        )
