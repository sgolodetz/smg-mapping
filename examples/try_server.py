import cv2

from smg.mapping import Server, UncompressedRGBDFrameDecoder


def main() -> None:
    server: Server = Server()
    server.start()

    client_id: int = 0
    decoder: UncompressedRGBDFrameDecoder = UncompressedRGBDFrameDecoder()
    while True:
        server.get_frame(client_id, decoder)
        cv2.imshow("Received RGB Image", decoder.get_rgb_image())
        cv2.imshow("Received Depth Image", decoder.get_depth_image() * 10)  # Multiply by 10 to make it visible
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
