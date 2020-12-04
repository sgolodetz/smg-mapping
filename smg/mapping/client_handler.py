import numpy as np
import socket
import threading

from typing import Callable, cast, List, Optional, Tuple, TypeVar

from smg.mapping import AckMessage, CalibrationMessage, FrameHeaderMessage, FrameMessage, Message, SocketUtil
from smg.utility import PooledQueue


# TYPE VARIABLE

T = TypeVar('T', bound=Message)


# MAIN CLASS

class ClientHandler:
    """Used to manage the connection to a client."""

    # CONSTRUCTOR

    def __init__(self, client_id: int, sock: socket.SocketType, should_terminate: threading.Event):
        """
        Construct a client handler.

        :param client_id:           The ID used by the server to refer to the client.
        :param sock:                The socket used to communicate with the client.
        :param should_terminate:    Whether or not the server should terminate (read-only, set within the server).
        """
        self.__calib_msg: Optional[CalibrationMessage] = None
        self.__client_id: int = client_id
        self.__connection_ok: bool = True
        self.__frame_message_queue: PooledQueue[FrameMessage] = PooledQueue[FrameMessage](PooledQueue.PES_DISCARD)
        self.__image_shapes: List[Tuple[int, int, int]] = []
        self.__intrinsics: List[Tuple[float, float, float, float]] = []
        self.__lock: threading.Lock = threading.Lock()
        self.__should_terminate: threading.Event = should_terminate
        self.__sock: socket.SocketType = sock
        self.__thread: Optional[threading.Thread] = None

    # PUBLIC METHODS

    def get_client_id(self) -> int:
        """
        Get the ID used by the server to refer to the client.

        :return:    The ID used by the server to refer to the client.
        """
        return self.__client_id

    def get_frame(self, receiver: Callable[[FrameMessage], None]) -> None:
        """
        TODO

        :param receiver:    TODO
        """
        with self.__lock:
            # Pass the first frame on the message queue to the frame receiver.
            receiver(self.__frame_message_queue.peek())

            # Pop the frame that's just been read from the message queue.
            self.__frame_message_queue.pop()

    def has_frames_now(self) -> bool:
        """
        Get whether or not the client is ready to yield a frame.

        :return:    True, if the client is ready to yield a frame, or False otherwise.
        """
        with self.__lock:
            return not self.__frame_message_queue.empty()

    def is_connection_ok(self) -> bool:
        """
        Get whether the connection is still ok (tracks whether or not the most recent read/write succeeded).

        :return:    True, if the connection is still ok, or False otherwise.
        """
        return self.__connection_ok

    def run_iter(self) -> None:
        """Run an iteration of the main loop for the client."""
        # Try to read a frame header message.
        header_msg: FrameHeaderMessage = FrameHeaderMessage(self.__calib_msg.get_max_images())
        self.__connection_ok = SocketUtil.read_message(self.__sock, header_msg)
        if self.__connection_ok:
            # If that succeeds, set up a frame message accordingly.
            image_shapes: List[Tuple[int, int, int]] = header_msg.extract_image_shapes()
            image_byte_sizes: List[int] = header_msg.extract_image_byte_sizes()
            frame_msg: FrameMessage = FrameMessage(image_shapes, image_byte_sizes)

            # Now, read the frame message itself.
            self.__connection_ok = SocketUtil.read_message(self.__sock, frame_msg)
            if self.__connection_ok:
                # TODO: Uncompression, eventually.
                with self.__frame_message_queue.begin_push() as push_handler:
                    elt: Optional[FrameMessage] = push_handler.get()
                    if elt is not None:
                        msg: FrameMessage = cast(FrameMessage, elt)
                        np.copyto(msg.get_data(), frame_msg.get_data())

                self.__connection_ok = SocketUtil.write_message(self.__sock, AckMessage())

    def run_post(self) -> None:
        """Run any code that should happen after the main loop for the client."""
        # Destroy the frame compressor prior to stopping the client handler.
        # TODO
        pass

    def run_pre(self) -> None:
        """Run any code that should happen before the main loop for the client."""
        # Read a calibration message from the client to get its camera intrinsics.
        self.__calib_msg = CalibrationMessage()
        self.__connection_ok = SocketUtil.read_message(self.__sock, self.__calib_msg)

        # If the calibration message was successfully read:
        if self.__connection_ok:
            # Save the camera parameters.
            self.__image_shapes = self.__calib_msg.extract_image_shapes()
            self.__intrinsics = self.__calib_msg.extract_intrinsics()

            # Print the camera parameters out for debugging purposes.
            print(
                f"Received camera parameters from client {self.__client_id}: {self.__image_shapes}, {self.__intrinsics}"
            )

            # Initialise the frame message queue.
            capacity: int = 5
            self.__frame_message_queue.initialise(capacity, lambda: FrameMessage(
                self.__calib_msg.extract_image_shapes(), self.__calib_msg.extract_uncompressed_image_byte_sizes()
            ))

            # Set up the frame compressor.
            # TODO

            # Construct a dummy frame message to consume messages that cannot be pushed onto the queue.
            # TODO

            # Signal to the client that the server is ready.
            self.__connection_ok = SocketUtil.write_message(self.__sock, AckMessage())

    def set_thread(self, thread: threading.Thread) -> None:
        """
        Set the thread that manages communication with the client.

        :param thread:  The thread that manages communication with the client.
        """
        self.__thread = thread
