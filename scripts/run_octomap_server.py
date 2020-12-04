import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Tuple

from smg.mapping import RGBDFrameReceiver, Server
from smg.pyoctomap import *
from smg.utility import ImageUtil


def main() -> None:
    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Octomap Server")

    # Set the projection matrix.
    glMatrixMode(GL_PROJECTION)
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)
    OctomapUtil.set_projection_matrix(intrinsics, *window_size)

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Set up the octree drawer.
    drawer: OcTreeDrawer = OcTreeDrawer()
    drawer.set_color_mode(CM_COLOR_HEIGHT)

    # Create the octree.
    voxel_size: float = 0.05
    tree: OcTree = OcTree(voxel_size)

    with Server() as server:
        server.start()

        client_id: int = 0
        pose: np.ndarray = np.eye(4)
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()

        while True:
            # Process any PyGame events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    tree.write_binary("remote_fusion.bt")
                    pygame.quit()
                    # noinspection PyProtectedMember
                    os._exit(0)

            if server.has_frames_now(client_id):
                # Get an RGB-D frame from the server.
                server.get_frame(client_id, receiver)
                pose = receiver.get_pose()

                # Use the depth image and pose to make an Octomap point cloud.
                pcd: Pointcloud = OctomapUtil.make_point_cloud(
                    ImageUtil.from_short_depth(receiver.get_depth_image()), pose, intrinsics
                )

                # Fuse the point cloud into the octree.
                start = timer()
                origin: Vector3 = Vector3(0.0, 0.0, 0.0)
                tree.insert_point_cloud(pcd, origin, discretize=True)
                end = timer()
                print(f"  - Time: {end - start}s")

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Draw the octree.
            OctomapUtil.draw_octree(tree, pose, drawer)

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
