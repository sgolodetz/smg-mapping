import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Optional

from smg.openni import OpenNICamera
from smg.mapping.remote import Client, RGBDFrameUtil
from smg.pyorbslam2 import RGBDTracker


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--use_tracker", action="store_true", help="whether to use the tracker"
    )
    args: dict = vars(parser.parse_args())

    tracker: Optional[RGBDTracker] = None
    if args["use_tracker"]:
        tracker = RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        )

    try:
        with OpenNICamera(mirror_images=True) as camera:
            with Client() as client:
                # Send a calibration message to tell the server the camera parameters.
                client.send_calibration_message(RGBDFrameUtil.make_calibration_message(
                    camera.get_colour_size(), camera.get_depth_size(),
                    camera.get_colour_intrinsics(), camera.get_depth_intrinsics()
                ))

                frame_idx: int = 0

                # Until the user wants to quit:
                while True:
                    # Grab an RGB-D pair from the camera.
                    rgb_image, depth_image = camera.get_images()
                    pose: Optional[np.ndarray] = None

                    # If we're using the tracker:
                    if tracker is not None:
                        # If the tracker's ready:
                        if tracker.is_ready():
                            # Try to estimate the pose of the camera.
                            inv_pose: np.ndarray = tracker.estimate_pose(rgb_image, depth_image)
                            if inv_pose is not None:
                                pose = np.linalg.inv(inv_pose)
                    else:
                        # Otherwise, simply use the identity matrix as a dummy pose.
                        pose = np.eye(4)

                    # If a pose is available (i.e. unless we were using the tracker and it failed):
                    if pose is not None:
                        # Send the RGB-D frame across to the server.
                        client.send_frame_message(lambda msg: RGBDFrameUtil.fill_frame_message(
                            frame_idx, rgb_image, depth_image, pose, msg
                        ))

                    # Show the RGB image so that the user can see what's going on (and exit if desired).
                    cv2.imshow("Kinect RGB-D Client", rgb_image)
                    c: int = cv2.waitKey(1)
                    if c == ord('q'):
                        break

                    # Increment the frame index.
                    frame_idx += 1
    except RuntimeError as e:
        print(e)
    finally:
        # If we're using the tracker:
        if tracker is not None:
            if tracker.is_ready():
                # If the tracker's ready, terminate it.
                tracker.terminate()
            else:
                # If the tracker's not ready yet, forcibly terminate the whole process (this isn't graceful, but
                # if we don't do it then we may have to wait a very long time for it to finish initialising).
                # noinspection PyProtectedMember
                os._exit(0)


if __name__ == "__main__":
    main()
