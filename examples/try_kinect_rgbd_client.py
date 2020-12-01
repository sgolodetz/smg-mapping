from typing import Tuple

from smg.mapping import CalibrationMessage, Client, SimpleMessage


def main() -> None:
    message: SimpleMessage[int] = SimpleMessage[int]()
    message.set_value(23)
    print(message.extract_value())
    # client: Client = Client()
    # calib_msg: CalibrationMessage = CalibrationMessage()
    # intrinsics: Tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0)
    # calib_msg.set_intrinsics(intrinsics)
    # print(calib_msg.extract_intrinsics())
    # client.send_calibration_message(calib_msg)


if __name__ == "__main__":
    main()
