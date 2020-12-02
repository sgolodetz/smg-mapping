import numpy as np

from typing import cast, Optional, Tuple

from smg.mapping import CalibrationMessage, Client, FrameMessage


def main() -> None:
    try:
        client: Client = Client()

        calib_msg: CalibrationMessage = CalibrationMessage()
        image_size: Tuple[int, int] = (480, 640)
        intrinsics: Tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0)
        calib_msg.set_image_size(image_size)
        calib_msg.set_intrinsics(intrinsics)
        print(calib_msg.extract_image_size(), calib_msg.extract_intrinsics())
        client.send_calibration_message(calib_msg)

        rgb_image: np.ndarray = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        depth_image: np.ndarray = np.zeros(image_size, dtype=np.float32)
        with client.begin_push_frame_message() as push_handler:
            elt: Optional[FrameMessage] = push_handler.get()
            if elt:
                msg: FrameMessage = cast(FrameMessage, elt)
                msg.set_frame_index(23)
                # TODO
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
