from typing import Tuple

from smg.mapping import CalibrationMessage, Client


def main() -> None:
    try:
        client: Client = Client()
        calib_msg: CalibrationMessage = CalibrationMessage()
        image_size: Tuple[int, int] = (640, 480)
        intrinsics: Tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0)
        calib_msg.set_image_size(image_size)
        calib_msg.set_intrinsics(intrinsics)
        print(calib_msg.extract_image_size(), calib_msg.extract_intrinsics())
        client.send_calibration_message(calib_msg)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
