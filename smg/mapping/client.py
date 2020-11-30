import socket

from typing import Tuple

from smg.mapping import CalibrationMessage


class Client:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, endpoint: Tuple[str, int] = ("127.0.0.1", 7851), *, timeout: int = 10):
        """
        TODO

        :param endpoint:    TODO
        :param timeout:     TODO
        """
        self.__sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__sock.connect(endpoint)
        self.__sock.settimeout(timeout)

    # PUBLIC METHODS

    def send_calibration_message(self, msg: CalibrationMessage) -> None:
        """
        TODO

        :param msg:     TODO
        """
        connection_ok: bool = True

        # TODO
        pass
