import cv2

from typing import cast, Optional

from smg.openni import OpenNICamera
from smg.mapping import Client, FrameMessage, RGBDFrameUtil


def main() -> None:
    try:
        with OpenNICamera(mirror_images=True) as camera:
            with Client() as client:
                # Send a calibration message to tell the server the camera parameters.
                client.send_calibration_message(RGBDFrameUtil.make_calibration_message(
                    camera.get_colour_size(), camera.get_depth_size(),
                    camera.get_colour_intrinsics(), camera.get_depth_intrinsics()
                ))

                # Until the user wants to quit:
                frame_idx: int = 0
                while True:
                    # Grab an RGB-D frame from the camera.
                    rgb_image, depth_image = camera.get_images()

                    # Send it across to the server.
                    with client.begin_push_frame_message() as push_handler:
                        elt: Optional[FrameMessage] = push_handler.get()
                        if elt:
                            msg: FrameMessage = cast(FrameMessage, elt)
                            RGBDFrameUtil.fill_frame_message(frame_idx, rgb_image, depth_image, msg)

                    # Show the RGB image so that the user can see what's going on (and exit if desired).
                    cv2.imshow("Sent RGB Image", rgb_image)
                    c: int = cv2.waitKey(1)
                    if c == ord('q'):
                        break

                    # Increment the frame index.
                    frame_idx += 1
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
