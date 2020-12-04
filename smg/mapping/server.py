from __future__ import annotations

import socket
import threading

from select import select
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

    # DESTRUCTOR

    def __del__(self):
        """Destroy the server."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """TODO"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """TODO"""
        self.terminate()

    # PUBLIC METHODS

    def get_frame(self, client_id: int, receiver: Callable[[FrameMessage], None]) -> None:
        """
        TODO

        :param client_id:   TODO
        :param receiver:    TODO
        """
        client_handler: ClientHandler = self._get_client_handler(client_id)
        if client_handler is not None:
            client_handler.get_frame(receiver)

    def has_finished(self, client_id: int) -> bool:
        """
        Get whether or not the specified client has finished.

        :param client_id:   The ID of the client to check.
        :return:            True, if the client has finished, or False otherwise.
        """
        with self.__lock:
            return client_id in self.__finished_clients

    def has_frames_now(self, client_id: int) -> bool:
        """
        Get whether or not the specified client is currently active and ready to yield a frame.

        :param client_id:   The ID of the client to check.
        :return:            True, if the client is currently active and ready to yield a frame, or False otherwise.
        """
        client_handler: ClientHandler = self._get_client_handler(client_id)
        return client_handler.has_frames_now() if client_handler else False

    def has_more_frames(self, client_id: int) -> bool:
        """
        Get whether or not the specified client is currently active and may still have more frames to yield.

        :param client_id:   The ID of the client to check.
        :return:            True, if the client is currently active and may still have more frames to yield,
                            or False otherwise.
        """
        return not self.has_finished(client_id)

    def start(self) -> None:
        """Start the server."""
        self.__server_thread.start()

    def terminate(self) -> None:
        """Tell the server to terminate."""
        with self.__lock:
            if not self.__should_terminate.is_set():
                self.__should_terminate.set()
                self.__server_thread.join()

    # PROTECTED METHODS

    def _get_client_handler(self, client_id: int) -> Optional[ClientHandler]:
        """
        Try to get the handler of the active client with the specified ID.

        .. note::
            If the server is still active and the client has not yet started, this will block.

        :param client_id:   The ID of the client whose handler we want to get.
        :return:            The client handler, if the client and the server are both active, or None otherwise.
        """
        with self.__lock:
            # Wait until one of the following is true:
            #   (i) The client is active
            #  (ii) The client has terminated
            # (iii) The server is terminating
            while self.__client_handlers.get(client_id) is None \
                    and client_id not in self.__finished_clients \
                    and not self.__should_terminate.is_set():
                self.__client_ready.wait(0.1)

            return self.__client_handlers.get(client_id)

    # PRIVATE METHODS

    def __handle_client(self, client_handler: ClientHandler) -> None:
        """
        Handle messages from a client.

        :param client_handler:  The handler for the client.
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

        # Once the client's finished, add it to the finished clients set and remove its handler.
        with self.__lock:
            print(f"Stopping client: {client_id}")
            self.__finished_clients.add(client_id)
            del(self.__client_handlers[client_id])
            print(f"Client terminated: {client_id}")

    def __run_server(self) -> None:
        """Run the server."""
        # Set up the server socket and listen for connections.
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", self.__port))
        server_sock.listen(5)

        print(f"Listening for connections on 127.0.0.1:{self.__port}...")

        while not self.__should_terminate.is_set():
            timeout: float = 0.1
            readable, _, _ = select([server_sock], [], [], timeout)
            if self.__should_terminate.is_set():
                break

            for s in readable:
                if s is server_sock:
                    client_sock, client_endpoint = server_sock.accept()
                    print(f"Accepted connection from client {self.__next_client_id} @ {client_endpoint}")
                    with self.__lock:
                        client_handler: ClientHandler = ClientHandler(
                            self.__next_client_id, client_sock, self.__should_terminate
                        )
                        client_thread: threading.Thread = threading.Thread(
                            target=self.__handle_client, args=[client_handler]
                        )
                        client_thread.start()
                        client_handler.set_thread(client_thread)
                        self.__next_client_id += 1
