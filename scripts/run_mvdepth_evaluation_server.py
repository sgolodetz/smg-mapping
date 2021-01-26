import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Optional, Tuple

from smg.mapping.remote import MappingServer, RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.mvdepthnet import MonocularDepthEstimator
from smg.utility import GeometryUtil, PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--pool_empty_strategy", "-p", type=str, default="wait",
        choices=("discard", "grow", "replace_random", "wait"),
        help="the strategy to use when a frame message is received whilst a client handler's frame pool is empty"
    )
    args: dict = vars(parser.parse_args())

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.EPoolEmptyStrategy.make(args["pool_empty_strategy"])
    ) as server:
        client_id: int = 0
        depth_estimator: MonocularDepthEstimator = MonocularDepthEstimator(
            "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"
        )
        frame_count: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        sum_density: float = 0.0
        sum_max_error: float = 0.0
        sum_mean_error: float = 0.0

        # Start the server.
        server.start()

        while True:
            # If the server has a frame from the client that has not yet been processed:
            if server.has_frames_now(client_id):
                # Get the camera intrinsics from the server, and pass them to the depth estimator.
                intrinsics: Tuple[float, float, float, float] = server.get_intrinsics(client_id)[0]
                depth_estimator.set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))

                # Get the frame from the server.
                server.get_frame(client_id, receiver)
                colour_image: np.ndarray = receiver.get_rgb_image()
                depth_image: np.ndarray = receiver.get_depth_image()
                tracker_w_t_c: np.ndarray = receiver.get_pose()

                # Try to estimate a depth image for the frame.
                estimated_depth_image: Optional[np.ndarray] = depth_estimator.estimate_depth(
                    colour_image, tracker_w_t_c
                )

                # If depth estimation was successful:
                if depth_image is not None and estimated_depth_image is not None:
                    # Show the ground truth and estimated depth images, as well as an L1 error image between them.
                    cv2.imshow("Depth Image", depth_image / 2)
                    cv2.imshow("Estimated Depth Image", estimated_depth_image / 2)

                    error_image: np.ndarray = np.abs(estimated_depth_image - depth_image)
                    valid_depths_mask: np.ndarray = np.where(
                        (depth_image > 0.0) & (estimated_depth_image > 0.0), 255, 0
                    ).astype(np.uint8)
                    error_image = np.where(valid_depths_mask > 0, error_image, 0.0)
                    cv2.imshow("L1 Error Image", error_image)

                    # Compute the depth estimation density, as well as the max and mean L1 errors.
                    density: float = 100 * np.count_nonzero(estimated_depth_image) / np.product(depth_image.shape)
                    max_error: float = np.max(error_image)
                    valid_depths_count: int = np.count_nonzero(valid_depths_mask)
                    mean_error: float = np.sum(error_image) / valid_depths_count if valid_depths_count > 0 else 0.0

                    # Update the sums of these metrics so that their averages can be calculated over the sequence.
                    sum_density += density
                    sum_max_error += max_error
                    sum_mean_error += mean_error

                    # Increment the frame count.
                    frame_count += 1

                    # Print out the computed metrics, as well as their averages over the sequence so far.
                    print(
                        f"Max Error: {max_error}; Mean Error: {mean_error}; Density: {density}%; "
                        f"Average Max Error: {sum_max_error / frame_count}; "
                        f"Average Mean Error: {sum_mean_error / frame_count}; "
                        f"Average Density: {sum_density / frame_count}%"
                    )

            # If we've ever seen a frame, update the OpenCV windows.
            if frame_count > 0:
                c: int = cv2.waitKey(1)

                # If the user presses 'q', exit.
                if c == ord('q'):
                    break


if __name__ == "__main__":
    main()
