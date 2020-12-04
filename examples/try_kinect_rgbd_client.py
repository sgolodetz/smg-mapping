import cv2
import struct

from typing import cast, Optional, Tuple

from smg.openni import OpenNICamera
from smg.mapping import CalibrationMessage, Client, FrameMessage, RGBDFrameUtil


def main() -> None:
    try:
        with OpenNICamera(mirror_images=True) as camera:
            with Client() as client:
                # Make and send a calibration message.
                calib_msg: CalibrationMessage = CalibrationMessage()
                # noinspection PyTypeChecker
                calib_msg.set_image_shapes([
                    camera.get_colour_size()[::-1] + (3,), camera.get_depth_size()[::-1] + (1,)
                ])
                calib_msg.set_intrinsics([camera.get_colour_intrinsics(), camera.get_depth_intrinsics()])
                calib_msg.set_pixel_byte_sizes([struct.calcsize("<B"), struct.calcsize("<H")])
                client.send_calibration_message(calib_msg)

                frame_idx: int = 0

                while True:
                    rgb_image, depth_image = camera.get_images()

                    with client.begin_push_frame_message() as push_handler:
                        elt: Optional[FrameMessage] = push_handler.get()
                        if elt:
                            msg: FrameMessage = cast(FrameMessage, elt)
                            RGBDFrameUtil.fill_frame_message(frame_idx, rgb_image, depth_image, msg)

                    frame_idx += 1

                    cv2.imshow("Sent RGB Image", rgb_image)
                    c: int = cv2.waitKey(1)
                    if c == ord('q'):
                        break
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
