import cv2

from smg.mapping.remote import MappingServer, RGBDFrameReceiver, RGBDFrameUtil


def main() -> None:
    with MappingServer(frame_decompressor=RGBDFrameUtil.decompress_frame_message) as server:
        server.start()

        client_id: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        seen_frame: bool = False

        while server.has_more_frames(client_id):
            if server.has_frames_now(client_id):
                server.get_frame(client_id, receiver)
                cv2.imshow("Received RGB Image", receiver.get_rgb_image())
                cv2.imshow("Received Depth Image", receiver.get_depth_image() * 10)  # Multiply by 10 to make it visible
                seen_frame = True

            if seen_frame:
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
