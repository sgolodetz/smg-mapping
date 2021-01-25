import cv2

from smg.mapping.remote import MappingServer, RGBDFrameMessageUtil, RGBDFrameReceiver


def main() -> None:
    with MappingServer(frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message) as server:
        client_id: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        seen_frame: bool = False

        # Start the server.
        server.start()

        # While the client hasn't terminated:
        while server.has_more_frames(client_id):
            # If an RGB-D frame is currently available:
            if server.has_frames_now(client_id):
                # Get it and show it. Scale the depth image to make it visible.
                server.get_frame(client_id, receiver)
                cv2.imshow("Received RGB Image", receiver.get_rgb_image())
                cv2.imshow("Received Depth Image", receiver.get_depth_image() / 2)
                seen_frame = True

            # If we've ever seen a frame:
            if seen_frame:
                # Run the OpenCV event loop, and break if the user presses 'q'.
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

        # Make sure all of the OpenCV windows are destroyed.
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
