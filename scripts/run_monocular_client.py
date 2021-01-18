import cv2
import numpy as np
import os

from argparse import ArgumentParser
from typing import Dict, Optional

from smg.imagesources import RGBFromRGBDImageSource, RGBImageSource
from smg.openni import OpenNICamera, OpenNIRGBDImageSource
from smg.mapping.remote import MappingClient, RGBDFrameMessageUtil
from smg.pyorbslam2 import MonocularTracker
from smg.rotory import DroneFactory, DroneRGBImageSource
from smg.utility import ImageUtil


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--source_type", "-t", type=str, required=True, choices=("ardrone2", "kinect", "tello"),
        help="the source type"
    )
    args: dict = vars(parser.parse_args())

    # noinspection PyUnusedLocal
    image_source: Optional[RGBImageSource] = None
    try:
        # Construct the RGB image source.
        # FIXME: This is duplicate code - factor it out.
        source_type: str = args["source_type"]
        if source_type == "kinect":
            image_source = RGBFromRGBDImageSource(OpenNIRGBDImageSource(OpenNICamera(mirror_images=True)))
        else:
            kwargs: Dict[str, dict] = {
                "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
                "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
            }
            image_source = DroneRGBImageSource(DroneFactory.make_drone(source_type, **kwargs[source_type]))

        # Construct the tracker.
        with MonocularTracker(
            settings_file=f"settings-{source_type}.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            # Construct the mapping client.
            with MappingClient(frame_compressor=RGBDFrameMessageUtil.compress_frame_message) as client:
                # Send a calibration message to tell the server the camera parameters.
                client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                    image_source.get_image_size(), image_source.get_image_size(),
                    image_source.get_intrinsics(), image_source.get_intrinsics()
                ))

                frame_idx: int = 0

                # Until the user wants to quit:
                while True:
                    # Grab an image from the image source.
                    image: np.ndarray = image_source.get_image()
                    pose: Optional[np.ndarray] = None

                    # If we're using the tracker:
                    if tracker is not None:
                        # If the tracker's ready:
                        if tracker.is_ready():
                            # Try to estimate the pose of the camera.
                            inv_pose: np.ndarray = tracker.estimate_pose(image)
                            if inv_pose is not None:
                                pose = np.linalg.inv(inv_pose)
                    else:
                        # Otherwise, simply use the identity matrix as a dummy pose.
                        pose = np.eye(4)

                    # If a pose is available (i.e. unless we were using the tracker and it failed):
                    if pose is not None:
                        # Send the frame across to the server.
                        dummy_depth_image: np.ndarray = np.zeros(image.shape[:2], dtype=np.float32)
                        client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                            frame_idx, image, ImageUtil.to_short_depth(dummy_depth_image), pose, msg
                        ))

                    # Show the image so that the user can see what's going on (and exit if desired).
                    cv2.imshow("Monocular Client", image)
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
