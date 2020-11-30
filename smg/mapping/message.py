import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple


class Message(ABC):
    """A message that can be sent across a network."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get_data(self) -> np.ndarray:
        """
        Get the message data.

        :return:    Get the message data.
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        """
        Get the size of the message.

        :return:    The size of the message.
        """
        pass

    # PROTECTED STATIC METHODS

    @staticmethod
    def _end_of(segment: Tuple[int, int]) -> int:
        """
        Get the end offset of the specified message segment.

        :param segment:     The segment.
        :return:            The end offset of the segment.
        """
        return segment[0] + segment[1]
