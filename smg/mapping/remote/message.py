import numpy as np

from abc import ABC
from typing import cast, Optional, Tuple


class Message(ABC):
    """A message that can be sent across a network."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a message."""
        self._data: Optional[np.ndarray] = None

    # PUBLIC METHODS

    def get_data(self) -> np.ndarray:
        """
        Get the message data.

        :return:    Get the message data.
        """
        return cast(np.ndarray, self._data)

    def get_size(self) -> int:
        """
        Get the size of the message.

        :return:    The size of the message.
        """
        return len(self.get_data())

    # PROTECTED METHODS

    def _data_for(self, segment: Tuple[int, int]) -> np.ndarray:
        """
        Get the data for the specified message segment.

        :param segment:     The segment.
        :return:            The data for the segment.
        """
        return self.get_data()[segment[0]:Message._end_of(segment)]

    # PROTECTED STATIC METHODS

    @staticmethod
    def _end_of(segment: Tuple[int, int]) -> int:
        """
        Get the end offset of the specified message segment.

        :param segment:     The segment.
        :return:            The end offset of the segment.
        """
        return segment[0] + segment[1]
