from .simple_message import SimpleMessage


class AckMessage(SimpleMessage[int]):
    """A message containing the acknowledgement for a previously received message."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct an acknowledgement message."""
        super().__init__(0, int)
