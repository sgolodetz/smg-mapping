import cv2
import numpy as np
import struct

from typing import cast, Optional

from smg.openni import OpenNICamera
from smg.mapping import CalibrationMessage, Client, FrameMessage
from smg.utility import ImageUtil


def main() -> None:
    try:
        with OpenNICamera(mirror_images=True) as camera:
            with Client() as client:
                # Send the calibration message.
                calib_msg: CalibrationMessage = CalibrationMessage()
                calib_msg.set_image_byte_sizes([
                    480 * 640 * 3 * struct.calcsize("<B"),
                    480 * 640 * struct.calcsize("<H")
                ])
                calib_msg.set_image_shapes([(480, 640, 3), (480, 640, 1)])
                calib_msg.set_intrinsics([camera.get_colour_intrinsics(), camera.get_depth_intrinsics()])
                client.send_calibration_message(calib_msg)

                frame_idx: int = 0

                while True:
                    rgb_image, depth_image = camera.get_images()

                    with client.begin_push_frame_message() as push_handler:
                        elt: Optional[FrameMessage] = push_handler.get()
                        if elt:
                            msg: FrameMessage = cast(FrameMessage, elt)
                            msg.set_frame_index(frame_idx)
                            msg.set_image_data(0, rgb_image.reshape(-1))
                            msg.set_image_data(1, ImageUtil.to_short_depth(depth_image).reshape(-1).view(np.uint8))

                    frame_idx += 1

                    cv2.imshow("Sent RGB Image", rgb_image)
                    c: int = cv2.waitKey(1)
                    if c == ord('q'):
                        break
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
