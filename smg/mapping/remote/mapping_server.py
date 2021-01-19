from __future__ import annotations

import socket
import threading

from select import select
from typing import Callable, Dict, Optional, List, Set, Tuple

from .mapping_client_handler import MappingClientHandler
from .frame_message import FrameMessage


class MappingServer:
    """A server that can be used to communicate with remote mapping clients."""

    # CONSTRUCTOR

    def __init__(self, port: int = 7851, *,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        """
        Construct a mapping server.

        :param port:                The port on which the server should listen for connections.
        :param frame_decompressor:  An optional function to use to decompress received frames.
        """
        self.__client_handlers: Dict[int, MappingClientHandler] = {}
        self.__finished_clients: Set[int] = set()
        self.__frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_decompressor
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
        """No-op (needed to allow the server's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the server at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def get_frame(self, client_id: int, receiver: Callable[[FrameMessage], None]) -> None:
        """
        Get the first frame from the specified client that has not yet been processed.

        .. note::
            The concept of a 'frame receiver' is used to obviate the server from needing to know about the contents
            of frame messages. This way, the frame receiver needs to know how to handle the frame message that it's
            given, but the server can just forward it to the receiver without caring.

        :param client_id:   The ID of the client.
        :param receiver:    The frame receiver to which to pass the first frame from the client that has not
                            yet been processed.
        """
        client_handler: MappingClientHandler = self._get_client_handler(client_id, wait_for_start=True)
        if client_handler is not None:
            client_handler.get_frame(receiver)

    def get_image_shapes(self, client_id: int) -> Optional[List[Tuple[int, int, int]]]:
        """
        Try to get the shapes of the images being produced by the different cameras being used by the specified client.

        :param client_id:   The ID of the client.
        :return:            The shapes of the images being produced by the different cameras, if the client
                            is active and a calibration message has been received from it, or None otherwise.
        """
        client_handler: MappingClientHandler = self._get_client_handler(client_id, wait_for_start=True)
        return client_handler.get_image_shapes() if client_handler is not None else None

    def get_intrinsics(self, client_id: int) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Try to get the intrinsics of the different cameras being used by the specified client.

        :param client_id:   The ID of the client.
        :return:            The intrinsics of the different cameras being used by the specified client
                            as a list of (fx,fy,cx,cy) tuples, if the client is active and a calibration
                            message has been received from it, or None otherwise.
        """
        client_handler: MappingClientHandler = self._get_client_handler(client_id, wait_for_start=True)
        return client_handler.get_intrinsics() if client_handler is not None else None

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
        client_handler: MappingClientHandler = self._get_client_handler(client_id, wait_for_start=False)
        return client_handler.has_frames_now() if client_handler is not None else False

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

    def _get_client_handler(self, client_id: int, *, wait_for_start: bool) -> Optional[MappingClientHandler]:
        """
        Try to get the handler of the active client with the specified ID.

        :param client_id:       The ID of the client whose handler we want to get.
        :param wait_for_start:  Whether or not to wait for the client to start, if it hasn't yet.
        :return:                The client handler, if the client and the server are both active, or None otherwise.
        """
        with self.__lock:
            # Wait until one of the following is true:
            #   (i) The client is active
            #  (ii) The client has terminated
            # (iii) The server is terminating
            while self.__client_handlers.get(client_id) is None \
                    and client_id not in self.__finished_clients \
                    and not self.__should_terminate.is_set():
                if wait_for_start:
                    self.__client_ready.wait(0.1)
                else:
                    break

            return self.__client_handlers.get(client_id)

    # PRIVATE METHODS

    def __handle_client(self, client_handler: MappingClientHandler) -> None:
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
                        client_handler: MappingClientHandler = MappingClientHandler(
                            self.__next_client_id, client_sock, self.__should_terminate,
                            frame_decompressor=self.__frame_decompressor
                        )
                        client_thread: threading.Thread = threading.Thread(
                            target=self.__handle_client, args=[client_handler]
                        )
                        client_thread.start()
                        client_handler.set_thread(client_thread)
                        self.__next_client_id += 1
