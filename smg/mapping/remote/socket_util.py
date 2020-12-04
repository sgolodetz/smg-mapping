import numpy as np
import socket

from typing import TypeVar

from .message import Message


# TYPE VARIABLE

T = TypeVar('T', bound=Message)


# MAIN CLASS

class SocketUtil:
    """Utility functions related to sockets."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def read_message(sock: socket.SocketType, msg: T) -> bool:
        """
        TODO

        :param sock:    TODO
        :param msg:     TODO
        :return:        TODO
        """
        try:
            data: bytes = b""
            while len(data) < msg.get_size():
                received: bytes = sock.recv(msg.get_size() - len(data))
                if len(received) > 0:
                    data += received
                else:
                    return False
            np.copyto(msg.get_data(), np.frombuffer(data, dtype=np.uint8))
            return True
        except (ConnectionResetError, socket.timeout, ValueError):
            return False

    @staticmethod
    def write_message(sock: socket.SocketType, msg: T) -> bool:
        """
        TODO

        :param sock:    TODO
        :param msg:     TODO
        :return:        TODO
        """
        sock.sendall(msg.get_data().tobytes())
        return True
