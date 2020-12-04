import numpy as np
import struct

from typing import List, Tuple

from smg.mapping import Message


class FrameMessage(Message):
    """A message containing a single frame of data (frame index + pose + images)."""

    # CONSTRUCTOR

    def __init__(self, image_shapes: List[Tuple[int, int, int]], image_byte_sizes: List[int]):
        """
        TODO

        :param image_shapes:        TODO
        :param image_byte_sizes:    TODO
        """
        super().__init__()

        self.__image_shapes: List[Tuple[int, int, int]] = image_shapes
        self.__image_byte_sizes: List[int] = image_byte_sizes

        self.__frame_index_fmt: str = "<i"
        self.__pose_byte_size: int = struct.calcsize("<ffffffffffffffff")
        self.__poses_fmt: str = "<" + "ffffffffffffffff" * len(image_shapes)

        self.__frame_index_segment: Tuple[int, int] = (
            0, struct.calcsize(self.__frame_index_fmt)
        )
        self.__poses_segment: Tuple[int, int] = (
            Message._end_of(self.__frame_index_segment), struct.calcsize(self.__poses_fmt)
        )
        self.__images_segment: Tuple[int, int] = (
            Message._end_of(self.__poses_segment), sum(self.__image_byte_sizes)
        )

        self._data = np.zeros(Message._end_of(self.__images_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def get_image_byte_sizes(self) -> List[int]:
        """
        TODO

        :return:    TODO
        """
        return self.__image_byte_sizes

    def get_image_data(self, image_idx: int) -> np.ndarray:
        """
        TODO

        :param image_idx:   TODO
        :return:            TODO
        """
        start: int = self.__images_segment[0]
        for i in range(image_idx):
            start += self.__image_byte_sizes[i]
        end: int = start + self.__image_byte_sizes[image_idx]
        return self._data[start:end]

    def get_image_shapes(self) -> List[Tuple[int, int, int]]:
        """
        TODO

        :return:    TODO
        """
        return self.__image_shapes

    def get_pose(self, image_idx: int) -> np.ndarray:
        """
        TODO

        :param image_idx:   TODO
        :return:            TODO
        """
        return self.__get_pose_data(image_idx).view(np.float32).reshape((4, 4))

    def set_frame_index(self, frame_index: int) -> None:
        """
        Copy a frame index into the appropriate byte segment in the message.

        :param frame_index: The frame index.
        """
        struct.pack_into(self.__frame_index_fmt, self._data, self.__frame_index_segment[0], frame_index)

    def set_image_data(self, image_idx: int, image_data: np.ndarray) -> None:
        """
        TODO

        :param image_idx:   TODO
        :param image_data:  TODO
        """
        np.copyto(self.get_image_data(image_idx), image_data)

    def set_pose(self, image_idx: int, pose: np.ndarray) -> None:
        """
        TODO

        :param image_idx:   TODO
        :param pose:        TODO
        """
        np.copyto(self.__get_pose_data(image_idx), pose.astype(np.float32).reshape(-1).view(np.uint8))

    # PRIVATE METHODS

    def __get_pose_data(self, image_idx: int) -> np.ndarray:
        """
        TODO

        :param image_idx:   TODO
        :return:            TODO
        """
        start: int = self.__poses_segment[0] + image_idx * self.__pose_byte_size
        return self._data[start:start+self.__pose_byte_size]
