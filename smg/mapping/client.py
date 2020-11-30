import socket

from typing import Tuple


class Client:
    """TODO"""

    def __init__(self, endpoint: Tuple[str, int] = ("127.0.0.1", 7851), *, timeout: int = 10):
        """
        TODO

        :param endpoint:    TODO
        :param timeout:     TODO
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(endpoint)
        sock.settimeout(timeout)
