import cv2

from smg.mapping import RGBDFrameReceiver, Server


def main() -> None:
    server: Server = Server()
    server.start()

    client_id: int = 0
    receiver: RGBDFrameReceiver = RGBDFrameReceiver()
    while True:
        server.get_frame(client_id, receiver)
        cv2.imshow("Received RGB Image", receiver.get_rgb_image())
        cv2.imshow("Received Depth Image", receiver.get_depth_image() * 10)  # Multiply by 10 to make it visible
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
