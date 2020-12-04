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
        Attempt to read a message of type T from the specified socket.

        :param sock:    The socket.
        :param msg:     The T into which to copy the message, if reading succeeded.
        :return:        True, if reading succeeded, or False otherwise.
        """
        try:
            data: bytes = b""

            # Until we've read the number of bytes we were expecting:
            while len(data) < msg.get_size():
                # Try to get the remaining bytes.
                received: bytes = sock.recv(msg.get_size() - len(data))

                # If we made progress, append the new bytes to the buffer. If not, something's wrong, so return False.
                if len(received) > 0:
                    data += received
                else:
                    return False

            # If we managed to get the number of bytes were were expecting, copy the buffer into the output message
            # and return True to indicate a successful read.
            np.copyto(msg.get_data(), np.frombuffer(data, dtype=np.uint8))
            return True
        except (ConnectionResetError, socket.timeout, ValueError):
            # If any exceptions are thrown during the read, return False.
            return False

    @staticmethod
    def write_message(sock: socket.SocketType, msg: T) -> bool:
        """
        Attempt to write a message of type T to the specified socket.

        :param sock:    The socket.
        :param msg:     The message.
        :return:        True, to allowing chaining of read and write calls.
        """
        # TODO: Investigate whether it's possible to detect when a write fails, and check for that if it is.
        sock.sendall(msg.get_data().tobytes())
        return True
