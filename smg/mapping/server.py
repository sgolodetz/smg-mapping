from __future__ import annotations

import socket
import threading

from typing import Dict

from smg.mapping import ClientHandler


class Server:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, port: int = 7851):
        """
        TODO

        :param port:    TODO
        """
        self.__client_handlers: Dict[int, ClientHandler] = {}
        self.__next_client_id: int = 0
        self.__port: int = port
        self.__server_thread: threading.Thread = threading.Thread(target=self.__run_server)
        self.__should_terminate: threading.Event = threading.Event()

        self.__lock: threading.Lock = threading.Lock()
        self.__client_ready: threading.Condition = threading.Condition(self.__lock)

    def start(self):
        """Start the server."""
        self.__server_thread.start()

    # PRIVATE METHODS

    def __handle_client(self, client_handler: ClientHandler) -> None:
        """
        TODO

        :param client_handler:  TODO
        """
        client_id: int = client_handler.get_client_id()
        print(f"Starting client: {client_id}")

        # Run the pre-loop code for the client.
        client_handler.run_pre()

        # Add the client handler to the dictionary of handlers for active clients.
        with self.__lock:
            self.__client_handlers[client_id] = client_handler

            # Signal to other threads that we're ready to start running the main loop for the client.
            print(f"Client ready: {client_id}")
            self.__client_ready.notify()

        # Run the main loop for the client. Loop until either (a) the connection drops, or (b) the server itself
        # is terminating.
        while client_handler.is_connection_ok() and not self.__should_terminate.is_set():
            client_handler.run_iter()

        # Run the post-loop code for the client.
        client_handler.run_post()

        # Once the client's finished, add it to the finished clients set so that it can be cleaned up.
        with self.__lock:
            print(f"Stopping client: {client_id}")
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
            with self.__lock:
                client_handler: ClientHandler = ClientHandler(
                    self.__next_client_id, client_sock, self.__should_terminate
                )
                client_thread: threading.Thread = threading.Thread(target=self.__handle_client, args=[client_handler])
                client_thread.start()
                client_handler.set_thread(client_thread)
                self.__next_client_id += 1
