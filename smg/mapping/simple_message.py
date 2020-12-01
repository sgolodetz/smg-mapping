import numpy as np
import pytypes
import struct

from typing import Generic, Optional, TypeVar

from smg.mapping import Message


# TYPE VARIABLE

T = TypeVar('T')


# MAIN CLASS

class SimpleMessage(Message, Generic[T]):
    """A message containing a single value of a specified type."""

    # CONSTRUCTOR

    def __init__(self, value: Optional[T] = None, t: Optional[type] = None):
        """
        Construct a simple message.

        .. note::
            It's not possible to infer t when deriving from SimpleMessage[T]. Subclasses must pass t in explicitly.

        :param value:   An optional initial message value.
        :param t:       The actual type of T (optional in some cases). If None, the class will try to infer it.
        """
        # noinspection PyUnusedLocal
        size: int = 0
        self.__fmt: str = ""

        if t is None:
            # TODO: This should be moved somewhere more central.
            # Try to get the actual type variable. Note that this is actually quite hard to do in Python, and so
            # we rely on some magic trickery in the latest version of pytypes, which can be installed via:
            #
            # python -m pip install git+https://github.com/sgolodetz/pytypes.git
            #
            # Importantly, this doesn't work for subclasses of SimpleMessage, which must pass in t explicitly.
            t = pytypes.type_util.get_orig_class(self).__args__[0]

        if t is int:
            self.__fmt = "i"
            size = 4
        else:
            raise RuntimeError(f"Cannot construct SimpleMessage with unsupported type {t.__name__}")

        self.__data: np.ndarray = np.zeros(size, dtype=np.uint8)

        if value is not None:
            self.set_value(value)

    # PUBLIC METHODS

    def extract_value(self) -> T:
        """
        Extract the message value.

        :return:    The message value.
        """
        return struct.unpack_from(self.__fmt, self.__data, 0)[0]

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

    def set_value(self, value: T) -> None:
        """
        Set the message value.

        :param value:   The message value.
        """
        struct.pack_into(self.__fmt, self.__data, 0, value)
