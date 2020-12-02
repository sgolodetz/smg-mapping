import cv2
import numpy as np
import time

from typing import cast, Optional, Tuple

from smg.mapping import CalibrationMessage, Client, FrameMessage


def main() -> None:
    try:
        with Client() as client:
            calib_msg: CalibrationMessage = CalibrationMessage()
            image_size: Tuple[int, int] = (480, 640)
            intrinsics: Tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0)
            calib_msg.set_image_size(image_size)
            calib_msg.set_intrinsics(intrinsics)
            print(calib_msg.extract_image_size(), calib_msg.extract_intrinsics())
            client.send_calibration_message(calib_msg)

            rgb_image: np.ndarray = cv2.imread("C:/smglib/smg-mapping/output-kinect/frame-000000.color.png")
            depth_image: np.ndarray = np.zeros(image_size, dtype=np.float32)
            with client.begin_push_frame_message() as push_handler:
                elt: Optional[FrameMessage] = push_handler.get()
                if elt:
                    msg: FrameMessage = cast(FrameMessage, elt)
                    msg.set_frame_index(23)
                    msg.set_rgb_image_data(rgb_image.reshape(-1))
                    # TODO

            time.sleep(1)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
