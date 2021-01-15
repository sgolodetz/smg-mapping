import cv2
import open3d as o3d
import os

from argparse import ArgumentParser
from typing import Optional

from smg.mapping import RGBDRemodeMapper
from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.openni import OpenNICamera, OpenNIRGBDImageSource
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import DepthEstimator, TemporalKeyframeDepthEstimator


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str,
        help="an optional directory into which to save the keyframes"
    )
    args: dict = vars(parser.parse_args())

    # noinspection PyUnusedLocal
    tsdf: Optional[o3d.pipelines.integration.ScalableTSDFVolume] = None

    with OpenNICamera(mirror_images=True) as camera:
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            depth_estimator: DepthEstimator = TemporalKeyframeDepthEstimator(
                camera.get_colour_size(), camera.get_colour_intrinsics(),
                denoising_iterations=400, max_images_per_keyframe=30
            )
            with RGBDRemodeMapper(
                OpenNIRGBDImageSource(camera), tracker, depth_estimator, output_dir=args.get("output_dir")
            ) as mapper:
                tsdf = mapper.run()

            # If ORB-SLAM's not ready yet, forcibly terminate the whole process (this isn't graceful, but
            # if we don't do it then we may have to wait a very long time for it to finish initialising).
            if not tracker.is_ready():
                # noinspection PyProtectedMember
                os._exit(0)

    cv2.destroyAllWindows()
    mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(tsdf, print_progress=True)
    VisualisationUtil.visualise_geometry(mesh)


if __name__ == "__main__":
    main()
