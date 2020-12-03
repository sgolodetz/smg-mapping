import matplotlib.pyplot as plt
import numpy as np

from smg.mapping import FrameMessage, Server
from smg.utility import ImageUtil


def show_colour_image(frame_msg: FrameMessage) -> None:
    rgb_image: np.ndarray = frame_msg.get_image_data(0).reshape(frame_msg.get_image_shapes()[0])
    depth_image: np.ndarray = frame_msg.get_image_data(1).view(np.uint16).reshape(frame_msg.get_image_shapes()[1][:2])
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(ImageUtil.flip_channels(rgb_image))
    ax[1].imshow(ImageUtil.from_short_depth(depth_image))
    plt.show()


def main() -> None:
    server: Server = Server()
    server.start()

    server.get_frame(0, show_colour_image)


if __name__ == "__main__":
    main()
