import socket
import threading

from typing import Optional, Tuple

from smg.mapping import AckMessage, CalibrationMessage, SocketUtil


class Client:
    """A client that can be used to communicate with a remote mapping server."""

    # CONSTRUCTOR

    def __init__(self, endpoint: Tuple[str, int] = ("127.0.0.1", 7851), *, timeout: int = 10):
        """
        Construct a client.

        :param endpoint:    The server host and port, e.g. ("127.0.0.1", 7851).
        :param timeout:     The socket timeout to use (in seconds).
        """
        self.__alive: bool = False
        self.__message_sender_thread: Optional[threading.Thread] = None

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
        if self.__alive:
            self.__sock.shutdown(socket.SHUT_RDWR)
            self.__sock.close()

    # PUBLIC METHODS

    def send_calibration_message(self, msg: CalibrationMessage) -> None:
        """
        Send a calibration message to the server.

        :param msg:     The message to send.
        """
        connection_ok: bool = True

        # Send the message to the server.
        connection_ok = connection_ok and SocketUtil.write_message(self.__sock, msg)

        # Wait for an acknowledgement (note that this is blocking, unless the connection fails).
        ack_msg: AckMessage = AckMessage()
        connection_ok = connection_ok and SocketUtil.read_message(self.__sock, ack_msg)

        # Throw if the message was not successfully sent and acknowledged.
        if not connection_ok:
            raise RuntimeError("Error: Failed to send calibration message")

        # Initialise the frame message queue.
        # TODO

        # Set up the frame compressor.
        # TODO

        # Start the message sender thread.
        self.__message_sender_thread = threading.Thread(target=self.__run_message_sender)
        self.__message_sender_thread.start()

    # PRIVATE METHODS

    def __run_message_sender(self) -> None:
        # TODO
        pass
