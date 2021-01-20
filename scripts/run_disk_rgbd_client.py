import cv2
import numpy as np
import os

from argparse import ArgumentParser

from smg.mapping.remote import MappingClient, RGBDFrameMessageUtil
from smg.utility import ImageUtil, PoseUtil


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
        with MappingClient(frame_compressor=RGBDFrameMessageUtil.compress_frame_message) as client:
            # Send a calibration message to tell the server the camera parameters.
            # client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
            #     camera.get_colour_size(), camera.get_depth_size(),
            #     camera.get_colour_intrinsics(), camera.get_depth_intrinsics()
            # ))

            frame_idx: int = 0

            # Until the user wants to quit:
            while True:
                # Try to load an RGB-D frame from disk.
                colour_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.color.png")
                depth_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.depth.png")
                pose_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.pose.txt")

                # If the colour image doesn't exist, early out.
                if not os.path.exists(colour_filename):
                    break

                colour_image: np.ndarray = cv2.imread(colour_filename)
                depth_image: np.ndarray = ImageUtil.load_depth_image(depth_filename)
                tracker_w_t_c: np.ndarray = np.linalg.inv(PoseUtil.load_pose(pose_filename))

                # Send the frame across to the server.
                # client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                #     frame_idx, colour_image, ImageUtil.to_short_depth(depth_image), tracker_w_t_c, msg
                # ))

                # Show the RGB image so that the user can see what's going on (and exit if desired).
                cv2.imshow("Disk RGB-D Client", colour_image)
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

                # Increment the frame index.
                frame_idx += 1
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
