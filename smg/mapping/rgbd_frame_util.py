import numpy as np

from smg.mapping import CalibrationMessage, FrameMessage
from smg.utility import ImageUtil


class RGBDFrameUtil:
    """TODO"""

    # PUBLIC STATIC METHODS

    @staticmethod
    def encode_frame_uncompressed(frame_idx: int, rgb_image: np.ndarray, depth_image: np.ndarray,
                                  msg: FrameMessage) -> None:
        """
        TODO

        :param frame_idx:   TODO
        :param rgb_image:   TODO
        :param depth_image: TODO
        :param msg:         TODO
        """
        msg.set_frame_index(frame_idx)
        msg.set_image_data(0, rgb_image.reshape(-1))
        msg.set_image_data(1, ImageUtil.to_short_depth(depth_image).reshape(-1).view(np.uint8))

    @staticmethod
    def make_calibration_message() -> CalibrationMessage:
        # TODO
        pass
