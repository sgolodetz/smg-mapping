import cv2
import numpy as np
import os

from typing import Optional

from smg.openni import OpenNICamera
from smg.mapping import Client, RGBDFrameUtil
from smg.pyorbslam2 import RGBDTracker


def main() -> None:
    try:
        with OpenNICamera(mirror_images=True) as camera:
            with RGBDTracker(
                    settings_file=f"settings-kinect.yaml", use_viewer=True,
                    voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
            ) as tracker:
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

                        # If the tracker's ready:
                        if tracker.is_ready():
                            # Try to estimate the pose of the camera.
                            pose: Optional[np.ndarray] = tracker.estimate_pose(rgb_image, depth_image)

                            # If this succeeds, send the RGB-D frame across to the server.
                            if pose is not None:
                                pose = np.linalg.inv(pose)
                                client.send_frame_message(lambda msg: RGBDFrameUtil.fill_frame_message(
                                    frame_idx, rgb_image, depth_image, pose, msg
                                ))

                        # Show the RGB image so that the user can see what's going on (and exit if desired).
                        cv2.imshow("Sent RGB Image", rgb_image)
                        c: int = cv2.waitKey(1)
                        if c == ord('q'):
                            break

                        # Increment the frame index.
                        frame_idx += 1

                    # If ORB-SLAM's not ready yet, forcibly terminate the whole process (this isn't graceful, but
                    # if we don't do it then we may have to wait a very long time for it to finish initialising).
                    if not tracker.is_ready():
                        # noinspection PyProtectedMember
                        os._exit(0)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
