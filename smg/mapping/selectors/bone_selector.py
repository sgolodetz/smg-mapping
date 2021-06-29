import cv2
import numpy as np

from typing import Optional

from smg.pyoctomap import OctomapPicker
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Keypoint, Skeleton3D


class BoneSelector:
    """A selector that uses the position and orientation of a bone in a skeleton to select 3D points."""

    # CONSTRUCTOR

    def __init__(self, skeleton: Skeleton3D, source_keypoint_name: str, target_keypoint_name: str):
        """
        Construct a bone selector.

        .. note::
            The bone is considered to define a half-ray starting at its "source" end and passing through its
            "target" end. The 3D point selected will be the point in the scene first hit by this half-ray.

        :param skeleton:                The skeleton.
        :param source_keypoint_name:    The name of the keypoint at the "source" end of the bone.
        :param target_keypoint_name:    The name of the keypoint at the "target" end of the bone.
        """
        self.__skeleton: Skeleton3D = skeleton
        self.__source_keypoint_name: str = source_keypoint_name
        self.__target_keypoint_name: str = target_keypoint_name

    # PUBLIC METHODS

    def get_selected_point(self, picker: OctomapPicker, *, debug: bool = False) -> Optional[np.ndarray]:
        """
        Get the currently selected 3D point (if any).

        :param picker:  The "picker", used to render a world-space points image of the scene from the perspective
                        of a camera looking along the bone.
        :param debug:   Whether to showing the picking image for debugging purposes.

        :return:    The currently selected 3D point (if any).
        """
        # Look up the bone's keypoints in the skeleton. If they're not present, early out.
        source_keypoint: Optional[Keypoint] = self.__skeleton.keypoints.get(self.__source_keypoint_name)
        target_keypoint: Optional[Keypoint] = self.__skeleton.keypoints.get(self.__target_keypoint_name)
        if source_keypoint is None or target_keypoint is None:
            return None

        # Construct a camera looking along the bone.
        up: np.ndarray = np.array([0.0, -1.0, 0.0])
        picking_cam: SimpleCamera = SimpleCamera(
            target_keypoint.position, target_keypoint.position - source_keypoint.position, up
        )
        picking_pose: np.ndarray = np.linalg.inv(CameraPoseConverter.camera_to_pose(picking_cam))

        # Use the picker to render a world-space points image from the perspective of this camera.
        picking_image, picking_mask = picker.pick(picking_pose)

        # If we're debugging, show the picking image.
        if debug:
            cv2.imshow("Picking Image", picking_image)
            cv2.waitKey(1)

        # If the world-space point at the centre of the picking image is valid, return that as the
        # selected point. Otherwise, return None.
        y, x = picking_image.shape[0] // 2, picking_image.shape[1] // 2
        return picking_image[y, x] if picking_mask[y, x] != 0 else None
