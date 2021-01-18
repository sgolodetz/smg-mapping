import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Optional

from smg.openni import OpenNICamera
from smg.mapping.remote import MappingClient, RGBDFrameMessageUtil
from smg.pyorbslam2 import RGBDTracker
from smg.utility import ImageUtil


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--suppress_tracker", action="store_true", help="whether to suppress the tracker"
    )
    args: dict = vars(parser.parse_args())

    # If requested, construct the tracker.
    tracker: Optional[RGBDTracker] = None
    if not args["suppress_tracker"]:
        tracker = RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=False,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        )

    try:
        with OpenNICamera(mirror_images=True) as camera:
            with MappingClient(frame_compressor=RGBDFrameMessageUtil.compress_frame_message) as client:
                # Send a calibration message to tell the server the camera parameters.
                client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
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

                    # Limit the depth range to 3m, since depth from the Kinect is less reliable beyond that.
                    depth_image = np.where(depth_image <= 3.0, depth_image, 0.0)

                    # If a pose is available (i.e. unless we were using the tracker and it failed):
                    if pose is not None:
                        # Send the RGB-D frame across to the server.
                        client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                            frame_idx, rgb_image, ImageUtil.to_short_depth(depth_image), pose, msg
                        ))

                    # Show the RGB image so that the user can see what's going on (and exit if desired).
                    cv2.imshow("Kinect RGB-D Client", rgb_image)
                    c: int = cv2.waitKey(1)
                    if c == ord('q'):
                        break

                    # Increment the frame index.
                    frame_idx += 1

            # Forcibly terminate the whole process. This isn't graceful, but both OpenNI and ORB-SLAM can
            # sometimes take a long time to shut down, and it's dull to wait for them. For example, if we
            # don't do this, then if we wanted to quit whilst the tracker was still initialising, we'd have
            # to sit around and wait for it to finish, as there's no way to cancel the initialisation.
            # noinspection PyProtectedMember
            os._exit(0)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
