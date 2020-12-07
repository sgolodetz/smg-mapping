import numpy as np
import struct

from typing import Generic, Optional, TypeVar

from smg.utility import TypeUtil

from .message import Message


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
            It's not possible to infer the actual type of T here when this constructor is invoked by a
            derived constructor. Instead, the derived constructor must pass it in explicitly.

        :param value:   An optional initial message value.
        :param t:       The actual type of T (optional in some cases). If None, the class will try to infer it.
        """
        super().__init__()

        # noinspection PyUnusedLocal
        size: int = 0
        self.__fmt: str = ""

        if t is None:
            # Try to get the actual type of T. Note that this approach, which relies on pytypes, won't work when
            # this constructor is invoked by a derived constructor, so subclasses of SimpleMessage must pass in
            # the type of T explicitly.
            t = TypeUtil.get_type_variable(self)

        if t is int:
            self.__fmt = "i"
            size = 4
        else:
            raise RuntimeError(f"Cannot construct SimpleMessage with unsupported type {t.__name__}")

        self._data = np.zeros(size, dtype=np.uint8)

        if value is not None:
            self.set_value(value)

    # PUBLIC METHODS

    def extract_value(self) -> T:
        """
        Extract the message value.

        :return:    The message value.
        """
        return struct.unpack_from(self.__fmt, self._data, 0)[0]

    def set_value(self, value: T) -> None:
        """
        Set the message value.

        :param value:   The message value.
        """
        struct.pack_into(self.__fmt, self._data, 0, value)
