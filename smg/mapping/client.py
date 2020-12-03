import socket
import threading

from typing import Callable, Optional, Tuple

from smg.mapping import AckMessage, CalibrationMessage, FrameHeaderMessage, FrameMessage, SocketUtil
from smg.utility import PooledQueue


class Client:
    """A client that can be used to communicate with a remote mapping server."""

    # CONSTRUCTOR

    def __init__(self, endpoint: Tuple[str, int] = ("127.0.0.1", 7851), *, timeout: int = 10,
                 frame_encoder: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        """
        Construct a client.

        :param endpoint:        The server host and port, e.g. ("127.0.0.1", 7851).
        :param timeout:         The socket timeout to use (in seconds).
        :param frame_encoder:   An optional function to use to encode frames prior to transmission.
        """
        self.__alive: bool = False
        self.__calib_msg: Optional[CalibrationMessage] = None
        self.__frame_encoder: Optional[Callable[[FrameMessage], FrameMessage]] = frame_encoder
        self.__frame_message_queue: PooledQueue[FrameMessage] = PooledQueue[FrameMessage](PooledQueue.PES_DISCARD)
        self.__message_sender_thread: Optional[threading.Thread] = None
        self.__should_terminate: threading.Event = threading.Event()

        try:
            self.__sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.connect(endpoint)
            self.__sock.settimeout(timeout)
            self.__alive = True
        except ConnectionRefusedError:
            raise RuntimeError("Error: Could not connect to the server")

    # DESTRUCTOR

    def __del__(self):
        """Destroy the client."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """TODO"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """TODO"""
        self.terminate()

    # PUBLIC METHODS

    def begin_push_frame_message(self) -> PooledQueue[FrameMessage].PushHandler:
        """
        Start the push of a frame message so that it can be sent to the server.

        :return:     A push handler that will handle the process of pushing a frame message onto the queue.
        """
        return self.__frame_message_queue.begin_push()

    def send_calibration_message(self, calib_msg: CalibrationMessage) -> None:
        """
        Send a calibration message to the server.

        :param calib_msg:   The calibration message.
        """
        connection_ok: bool = True

        # Send the message to the server.
        connection_ok = connection_ok and SocketUtil.write_message(self.__sock, calib_msg)

        # Wait for an acknowledgement (note that this is blocking, unless the connection fails).
        ack_msg: AckMessage = AckMessage()
        connection_ok = connection_ok and SocketUtil.read_message(self.__sock, ack_msg)

        # If the meessage was successfully sent and acknowledged, save it, else throw.
        if connection_ok:
            self.__calib_msg = calib_msg
        else:
            raise RuntimeError("Error: Failed to send calibration message")

        # Initialise the frame message queue.
        capacity: int = 1
        self.__frame_message_queue.initialise(capacity, lambda: FrameMessage(
            calib_msg.extract_image_shapes(), calib_msg.extract_image_byte_sizes()
        ))

        # Set up the frame compressor.
        # TODO

        # Start the message sender thread.
        self.__message_sender_thread = threading.Thread(target=self.__run_message_sender)
        self.__message_sender_thread.start()

    def terminate(self) -> None:
        """Tell the client to terminate."""
        if self.__alive:
            self.__should_terminate.set()
            self.__message_sender_thread.join()
            self.__sock.shutdown(socket.SHUT_RDWR)
            self.__sock.close()
            self.__alive = False

    # PRIVATE METHODS

    def __run_message_sender(self) -> None:
        """Send frame messages from the message queue across to the server."""
        ack_msg: AckMessage = AckMessage()

        connection_ok: bool = True

        while connection_ok and not self.__should_terminate.is_set():
            # Try to read the first frame message from the queue (this will block until a message is available,
            # except when the termination flag is set, in which case it will return None).
            frame_msg: Optional[FrameMessage] = self.__frame_message_queue.peek(self.__should_terminate)

            # If the termination flag is set, exit.
            if self.__should_terminate.is_set():
                break

            # If requested, encode the frame prior to transmission.
            encoded_frame_msg: FrameMessage = frame_msg
            if self.__frame_encoder is not None:
                encoded_frame_msg = self.__frame_encoder(frame_msg)

            # Make the frame header message.
            header_msg: FrameHeaderMessage = FrameHeaderMessage(self.__calib_msg.get_max_images())
            header_msg.set_image_byte_sizes(encoded_frame_msg.get_image_byte_sizes())
            header_msg.set_image_shapes(encoded_frame_msg.get_image_shapes())

            # First send the frame header message, then send the frame message, then wait for an acknowledgement
            # from the server. We chain all of these with 'and' so as to early out in case of failure.
            connection_ok = connection_ok and \
                SocketUtil.write_message(self.__sock, header_msg) and \
                SocketUtil.write_message(self.__sock, encoded_frame_msg) and \
                SocketUtil.read_message(self.__sock, ack_msg)

            # Remove the frame message that we have just sent from the queue.
            self.__frame_message_queue.pop()
