import socket
import threading

from typing import Optional

from smg.mapping import Message


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
        pass

    def set_thread(self, thread: threading.Thread) -> None:
        """
        TODO

        :param thread:  TODO
        """
        self.__thread = thread

    # PRIVATE METHODS

    def __read_message(self, msg: Message) -> None:
        # TODO
        pass
