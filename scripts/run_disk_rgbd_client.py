import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Any, Dict, Optional

from smg.mapping.remote import MappingClient, RGBDFrameMessageUtil
from smg.utility import CameraParameters, ImageUtil, PooledQueue, RGBDSequenceUtil


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the directory from which to load the sequence"
    )
    args: dict = vars(parser.parse_args())

    sequence_dir: str = args["sequence_dir"]

    try:
        with MappingClient(
            frame_compressor=RGBDFrameMessageUtil.compress_frame_message,
            pool_empty_strategy=PooledQueue.PES_GROW
        ) as client:
            # Try to load the camera parameters for the sequence. If this fails, raise an exception.
            calib: Optional[CameraParameters] = RGBDSequenceUtil.try_load_calibration(sequence_dir)
            if calib is None:
                raise RuntimeError(f"Cannot load calibration from '{sequence_dir}'")

            # Send a calibration message to tell the server the camera parameters.
            client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                calib.get_image_size("colour"), calib.get_image_size("depth"),
                calib.get_intrinsics("colour"), calib.get_intrinsics("depth")
            ))

            frame_idx: int = 0
            colour_image: Optional[np.ndarray] = None

            # Until the user wants to quit:
            while True:
                # Try to load an RGB-D frame from disk.
                frame: Optional[Dict[str, Any]] = RGBDSequenceUtil.try_load_frame(frame_idx, sequence_dir)

                # If the frame was successfully loaded:
                if frame is not None:
                    # Send it across to the server.
                    client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                        frame_idx,
                        frame["colour_image"],
                        ImageUtil.to_short_depth(frame["depth_image"]),
                        frame["world_from_camera"],
                        msg
                    ))

                    # Increment the frame index.
                    frame_idx += 1

                    # Update the colour image so that it can be shown.
                    colour_image = frame["colour_image"]

                # Show the most recent colour image (if any) so that the user can see what's going on.
                if colour_image is not None:
                    cv2.imshow("Disk RGB-D Client", colour_image)
                    c: int = cv2.waitKey(1)
                    if c == ord('q'):
                        break
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
