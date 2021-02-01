import numpy as np

from argparse import ArgumentParser
from typing import Optional

from smg.mapping import MVDepthOctomapMappingSystem
from smg.mapping.remote import MappingServer, RGBDFrameMessageUtil
from smg.mvdepthnet import MonocularDepthEstimator
from smg.utility import PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", "-o", type=str,
        help="an optional directory into which to save the sequence"
    )
    parser.add_argument(
        "--pool_empty_strategy", "-p", type=str, default="discard",
        choices=("discard", "grow", "replace_random", "wait"),
        help="the strategy to use when a frame message is received whilst a client handler's frame pool is empty"
    )
    args: dict = vars(parser.parse_args())

    output_dir: Optional[str] = args["output_dir"]

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.EPoolEmptyStrategy.make(args["pool_empty_strategy"])
    ) as server:
        # Construct the depth estimator.
        depth_estimator: MonocularDepthEstimator = MonocularDepthEstimator(
            "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar", debug=True
        )

        # Construct the mapping system.
        mapping_system: MVDepthOctomapMappingSystem = MVDepthOctomapMappingSystem(
            depth_estimator=depth_estimator, output_dir=output_dir, server=server
        )

        # Start the server.
        server.start()

        # Run the mapping system.
        mapping_system.run()


if __name__ == "__main__":
    main()
