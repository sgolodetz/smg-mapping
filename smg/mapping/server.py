from __future__ import annotations

import socket
import threading

from typing import Callable, Dict, Optional, Set

from smg.mapping import ClientHandler, FrameMessage


class Server:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, port: int = 7851):
        """
        TODO

        :param port:    TODO
        """
        self.__client_handlers: Dict[int, ClientHandler] = {}
        self.__finished_clients: Set[int] = set()
        self.__next_client_id: int = 0
        self.__port: int = port
        self.__server_thread: threading.Thread = threading.Thread(target=self.__run_server)
        self.__should_terminate: threading.Event = threading.Event()

        self.__lock: threading.Lock = threading.Lock()
        self.__client_ready: threading.Condition = threading.Condition(self.__lock)

    # PUBLIC METHODS

    def get_frame(self, client_id: int, decoder: Callable[[FrameMessage], None]) -> None:
        """
        TODO

        :param client_id:   TODO
        :param decoder:     TODO
        """
        # Look up the handler for the client whose frame we want to get. If the client is no longer active, early out.
        client_handler = self._get_client_handler(client_id)
        if client_handler is None:
            return

        # Pass the first frame on the client's message queue to the frame decoder.
        decoder(client_handler.get_frame_message_queue().peek())

        # Pop the frame that's just been read from the message queue.
        client_handler.get_frame_message_queue().pop()

    def start(self):
        """Start the server."""
        self.__server_thread.start()

    # PROTECTED METHODS

    def _get_client_handler(self, client_id: int) -> Optional[ClientHandler]:
        """
        Try to get the handler of the active client with the specified ID.

        .. note::
            If the client has not yet started, this will block.

        :param client_id:   The ID of the client whose handler we want to get.
        :return:            The client handler, if the client is active, or None if it has finished.
        """
        with self.__lock:
            # Wait until the client is either active or has terminated.
            while self.__client_handlers.get(client_id) is None and client_id not in self.__finished_clients:
                self.__client_ready.wait(0.1)

            return self.__client_handlers.get(client_id)

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
        # TODO: Better handle what happens when the connection drops during the pre-loop code.

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
            self.__finished_clients.add(client_id)
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
