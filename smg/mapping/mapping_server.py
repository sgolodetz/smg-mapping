from __future__ import annotations

import socket
import threading

from typing import Optional

from smg.pyoctomap import *


class MappingServer:
    """TODO"""

    # NESTED TYPES

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

        def set_thread(self, thread: threading.Thread) -> None:
            """
            TODO

            :param thread:  TODO
            """
            self.__thread = thread

    # CONSTRUCTOR

    def __init__(self, port: int = 7851):
        """
        TODO

        :param port:    TODO
        """
        self.__next_client_id: int = 0
        self.__port: int = port
        self.__server_thread: threading.Thread = threading.Thread(target=self.__run_server)
        self.__should_terminate: threading.Event = threading.Event()

    def start(self):
        """Start the server."""
        self.__server_thread.start()

    # PRIVATE METHODS

    def __handle_client(self, client_handler: MappingServer.ClientHandler) -> None:
        """
        TODO

        :param client_handler:  TODO
        """
        client_id: int = client_handler.get_client_id()
        print(f"Starting client {client_id}")

        # TODO

    def __run_server(self) -> None:
        """TODO"""
        # Set up the server socket and listen for connections.
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", self.__port))
        server_sock.listen(5)

        print(f"Listening for connections on 127.0.0.1:{self.__port}...")

        while not self.__should_terminate.is_set():
            client_sock, client_endpoint = server_sock.accept()
            print(f"Accepted connection from client {self.__next_client_id} @ {client_endpoint}")
            client_handler: MappingServer.ClientHandler = MappingServer.ClientHandler(
                self.__next_client_id, client_sock, self.__should_terminate
            )
            self.__next_client_id += 1
            client_thread: threading.Thread = threading.Thread(target=self.__handle_client, args=[client_handler])
            client_handler.set_thread(client_thread)
            client_thread.start()
