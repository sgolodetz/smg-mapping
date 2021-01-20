import numpy as np
import socket
import threading

from typing import Callable, cast, List, Optional, Tuple, TypeVar

from smg.utility import PooledQueue

from .ack_message import AckMessage
from .calibration_message import CalibrationMessage
from .frame_header_message import FrameHeaderMessage
from .frame_message import FrameMessage
from .message import Message
from .socket_util import SocketUtil


# TYPE VARIABLE

T = TypeVar('T', bound=Message)


# MAIN CLASS

class MappingClientHandler:
    """Used to manage the connection to a mapping client."""

    # CONSTRUCTOR

    def __init__(self, client_id: int, sock: socket.SocketType, should_terminate: threading.Event, *,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        """
        Construct a mapping client handler.

        :param client_id:           The ID used by the server to refer to the client.
        :param sock:                The socket used to communicate with the client.
        :param should_terminate:    Whether or not the server should terminate (read-only, set within the server).
        :param frame_decompressor:  An optional function to use to decompress received frames.
        """
        self.__calib_msg: Optional[CalibrationMessage] = None
        self.__client_id: int = client_id
        self.__connection_ok: bool = True
        self.__frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_decompressor
        self.__frame_message_queue: PooledQueue[FrameMessage] = PooledQueue[FrameMessage](PooledQueue.PES_DISCARD)
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
        Get the first frame from the client that has not yet been processed.

        .. note::
            The concept of a 'frame receiver' is used to obviate the client handler from needing to know about
            the contents of frame messages. This way, the frame receiver needs to know how to handle the frame
            message that it's given, but the client handler can just forward it to the receiver without caring.

        :param receiver:    The frame receiver to which to pass the first frame from the client that has not
                            yet been processed.
        """
        with self.__lock:
            # Pass the first frame on the message queue to the frame receiver.
            receiver(self.__frame_message_queue.peek())

            # Pop the frame that's just been read from the message queue.
            self.__frame_message_queue.pop()

    def get_image_shapes(self) -> Optional[List[Tuple[int, int, int]]]:
        """
        Try to get the shapes of the images being produced by the different cameras being used.

        :return:    The shapes of the images being produced by the different cameras, if a calibration
                    message has been received from the client, or None otherwise.
        """
        return self.__calib_msg.get_image_shapes() if self.__calib_msg is not None else None

    def get_intrinsics(self) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Try to get the intrinsics of the different cameras being used.

        :return:    The intrinsics of the different cameras being used, as (fx, fy, cx, cy) tuples,
                    if a calibration message has been received from the client, or None otherwise.
        """
        return self.__calib_msg.get_intrinsics() if self.__calib_msg is not None else None

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

        # If that succeeds:
        if self.__connection_ok:
            # Set up a frame message accordingly.
            image_shapes: List[Tuple[int, int, int]] = header_msg.get_image_shapes()
            image_byte_sizes: List[int] = header_msg.get_image_byte_sizes()
            frame_msg: FrameMessage = FrameMessage(image_shapes, image_byte_sizes)

            # Try to read the contents of the frame message from the client.
            self.__connection_ok = SocketUtil.read_message(self.__sock, frame_msg)

            # If that succeeds:
            if self.__connection_ok:
                # Decompress the frame as necessary.
                decompressed_frame_msg: FrameMessage = frame_msg
                if self.__frame_decompressor is not None:
                    decompressed_frame_msg = self.__frame_decompressor(frame_msg)

                # Push the decompressed frame onto the message queue.
                with self.__frame_message_queue.begin_push() as push_handler:
                    elt: Optional[FrameMessage] = push_handler.get()
                    if elt is not None:
                        msg: FrameMessage = cast(FrameMessage, elt)
                        np.copyto(msg.get_data(), decompressed_frame_msg.get_data())

                # Send an acknowledgement to the client.
                self.__connection_ok = SocketUtil.write_message(self.__sock, AckMessage())

    def run_post(self) -> None:
        """Run any code that should happen after the main loop for the client."""
        # This is currently a no-op.
        pass

    def run_pre(self) -> None:
        """Run any code that should happen before the main loop for the client."""
        # Read a calibration message from the client.
        self.__calib_msg = CalibrationMessage()
        self.__connection_ok = SocketUtil.read_message(self.__sock, self.__calib_msg)

        # If the calibration message was successfully read:
        if self.__connection_ok:
            # Print the camera parameters out for debugging purposes.
            image_shapes: List[Tuple[int, int, int]] = self.__calib_msg.get_image_shapes()
            intrinsics: List[Tuple[float, float, float, float]] = self.__calib_msg.get_intrinsics()
            print(
                f"Received camera parameters from client {self.__client_id}: {image_shapes}, {intrinsics}"
            )

            # Initialise the frame message queue.
            capacity: int = 5
            self.__frame_message_queue.initialise(capacity, lambda: FrameMessage(
                self.__calib_msg.get_image_shapes(), self.__calib_msg.get_uncompressed_image_byte_sizes()
            ))

            # Signal to the client that the server is ready.
            self.__connection_ok = SocketUtil.write_message(self.__sock, AckMessage())

    def set_thread(self, thread: threading.Thread) -> None:
        """
        Set the thread that manages communication with the client.

        :param thread:  The thread that manages communication with the client.
        """
        self.__thread = thread
