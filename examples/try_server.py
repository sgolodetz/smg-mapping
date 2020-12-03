import cv2
import matplotlib.pyplot as plt
import numpy as np

from smg.mapping import FrameMessage, Server
from smg.utility import ImageUtil


_, ax = plt.subplots(1, 2)


def show_colour_image(frame_msg: FrameMessage) -> None:
    rgb_image: np.ndarray = frame_msg.get_image_data(0).reshape(frame_msg.get_image_shapes()[0])
    depth_image: np.ndarray = frame_msg.get_image_data(1).view(np.uint16).reshape(frame_msg.get_image_shapes()[1][:2])
    # ax[0].clear()
    # ax[1].clear()
    # ax[0].imshow(ImageUtil.flip_channels(rgb_image))
    # ax[1].imshow(ImageUtil.from_short_depth(depth_image))
    # plt.draw()
    # plt.waitforbuttonpress(0.001)
    cv2.imshow("Received Image", rgb_image)
    cv2.waitKey(1)


def main() -> None:
    server: Server = Server()
    server.start()

    client_id: int = 0
    while True:
        server.get_frame(client_id, show_colour_image)


if __name__ == "__main__":
    main()
