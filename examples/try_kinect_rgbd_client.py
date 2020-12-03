import cv2
import numpy as np
import struct
import time

from typing import cast, Optional, Tuple

from smg.mapping import CalibrationMessage, Client, FrameMessage
from smg.utility import ImageUtil


def main() -> None:
    try:
        with Client() as client:
            calib_msg: CalibrationMessage = CalibrationMessage()
            intrinsics: Tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0)
            calib_msg.set_image_byte_sizes([
                480 * 640 * 3 * struct.calcsize("<B"),
                480 * 640 * struct.calcsize("<H")
            ])
            calib_msg.set_image_shapes([(480, 640, 3), (480, 640, 1)])
            calib_msg.set_intrinsics([intrinsics, intrinsics])
            print(calib_msg.extract_image_shapes(), calib_msg.extract_intrinsics())
            client.send_calibration_message(calib_msg)

            rgb_image: np.ndarray = cv2.imread("C:/smglib/smg-mapping/output-kinect/frame-000000.color.png")
            depth_image: np.ndarray = ImageUtil.to_short_depth(
                ImageUtil.load_depth_image("C:/smglib/smg-mapping/output-kinect/frame-000000.depth.png")
            )
            with client.begin_push_frame_message() as push_handler:
                elt: Optional[FrameMessage] = push_handler.get()
                if elt:
                    msg: FrameMessage = cast(FrameMessage, elt)
                    msg.set_frame_index(23)
                    msg.set_image_data(0, rgb_image.reshape(-1))
                    msg.set_image_data(1, depth_image.reshape(-1).view(np.uint8))

            time.sleep(1)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
