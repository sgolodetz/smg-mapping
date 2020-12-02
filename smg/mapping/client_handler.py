import socket
import threading

from typing import Optional, Tuple, TypeVar

from smg.mapping import AckMessage, CalibrationMessage, Message, SocketUtil


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
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
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
        """Run an iteration of the main loop for the client."""
        pass

    def run_post(self) -> None:
        """Run any code that should happen after the main loop for the client."""
        # Destroy the frame compressor prior to stopping the client handler.
        # TODO
        pass

    def run_pre(self) -> None:
        """Run any code that should happen before the main loop for the client."""
        # Read a calibration message from the client to get its camera intrinsics.
        calib_msg: CalibrationMessage = CalibrationMessage()
        self.__connection_ok = SocketUtil.read_message(self.__sock, calib_msg)

        # If the calibration message was successfully read:
        if self.__connection_ok:
            # Save the camera intrinsics.
            self.__intrinsics = calib_msg.extract_intrinsics()

            # Print the intrinsics out for debugging purposes.
            print(f"Received camera intrinsics from client {self.__client_id}: {self.__intrinsics}")

            # Initialise the frame message queue.
            capacity: int = 5
            # TODO

            # Set up the frame compressor.
            # TODO

            # Construct a dummy frame message to consume messages that cannot be pushed onto the queue.
            # TODO

            # Signal to the client that the server is ready.
            self.__connection_ok = SocketUtil.write_message(self.__sock, AckMessage())

    def set_thread(self, thread: threading.Thread) -> None:
        """
        TODO

        :param thread:  TODO
        """
        self.__thread = thread
