import cv2
import open3d as o3d
import os

from argparse import ArgumentParser
from typing import Dict, Optional

from smg.imagesources import RGBFromRGBDImageSource, RGBImageSource
from smg.mapping import MonocularMappingSystem
from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.openni import OpenNICamera, RGBDOpenNICamera
from smg.pyorbslam2 import MonocularTracker
from smg.pyremode import DepthEstimator, TemporalKeyframeDepthEstimator
from smg.rotory.drone_factory import DroneFactory
from smg.rotory.rgb_drone_camera import RGBDroneCamera


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str,
        help="an optional directory into which to save the keyframes"
    )
    parser.add_argument(
        "--source_type", "-t", type=str, required=True, choices=("ardrone2", "kinect", "tello"),
        help="the source type"
    )
    args: dict = vars(parser.parse_args())

    # noinspection PyUnusedLocal
    tsdf: Optional[o3d.pipelines.integration.ScalableTSDFVolume] = None
    image_source: Optional[RGBImageSource] = None
    try:
        # Construct the RGB image source.
        # FIXME: This is duplicate code - factor it out.
        source_type: str = args["source_type"]
        if source_type == "kinect":
            image_source = RGBFromRGBDImageSource(RGBDOpenNICamera(OpenNICamera(mirror_images=True)))
        else:
            kwargs: Dict[str, dict] = {
                "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
                "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
            }
            image_source = RGBDroneCamera(DroneFactory.make_drone(source_type, **kwargs[source_type]))

        # Construct the depth estimator.
        depth_estimator: DepthEstimator = TemporalKeyframeDepthEstimator(
            image_source.get_image_size(), image_source.get_intrinsics(),
            denoising_iterations=400
        )

        # Run the mapping system.
        with MonocularTracker(
            settings_file=f"settings-{source_type}.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            with MonocularMappingSystem(
                image_source, tracker, depth_estimator, output_dir=args.get("output_dir")
            ) as system:
                tsdf = system.run()

            # If ORB-SLAM's not ready yet, forcibly terminate the whole process (this isn't graceful, but
            # if we don't do it then we may have to wait a very long time for it to finish initialising).
            if not tracker.is_ready():
                # noinspection PyProtectedMember
                os._exit(0)
    finally:
        # Terminate the image source once we've finished mapping.
        if image_source is not None:
            image_source.terminate()

        # Close any remaining OpenCV windows.
        cv2.destroyAllWindows()

    # Show the reconstructed map.
    mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(tsdf, print_progress=True)
    VisualisationUtil.visualise_geometry(mesh)


if __name__ == "__main__":
    main()
