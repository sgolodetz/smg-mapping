import numpy as np
import socket
import threading

from typing import Optional, TypeVar

from smg.mapping import CalibrationMessage, Message


# TYPE VARIABLE

T = TypeVar('T', bound=Message)


# MAIN CLASS

class ClientHandler:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, client_id: int, sock: socket.SocketType, should_terminate: threading.Event):
        """
        TODO

        :param client_id:           TODO
        :param sock:                TODO
        :param should_terminate:    TODO
        """
        self.__client_id: int = client_id
        self.__connection_ok: bool = True
        self.__should_terminate: threading.Event = should_terminate
        self.__sock: socket.SocketType = sock
        self.__thread: Optional[threading.Thread] = None

    # PUBLIC METHODS

    def get_client_id(self) -> int:
        """
        TODO

        :return:    TODO
        """
        return self.__client_id

    def is_connection_ok(self) -> bool:
        """
        TODO

        :return:    TODO
        """
        return self.__connection_ok

    def run_iter(self) -> None:
        """TODO"""
        pass

    def run_post(self) -> None:
        """TODO"""
        pass

    def run_pre(self) -> None:
        """TODO"""
        # Read a calibration message from the client to get its camera intrinsics.
        calib_msg: CalibrationMessage = CalibrationMessage()
        self.__connection_ok = self.__read_message(calib_msg)

        # If the calibration message was successfully read:
        if self.__connection_ok:
            # TODO
            print(calib_msg.extract_intrinsics())

    def set_thread(self, thread: threading.Thread) -> None:
        """
        TODO

        :param thread:  TODO
        """
        self.__thread = thread

    # PRIVATE METHODS

    def __read_message(self, msg: T) -> bool:
        """
        TODO

        :param msg: TODO
        :return:    TODO
        """
        try:
            data = self.__sock.recv(msg.get_size())
            np.copyto(msg.get_data(), np.frombuffer(data, dtype=np.uint8))
            return True
        except (ConnectionResetError, ValueError):
            return False
