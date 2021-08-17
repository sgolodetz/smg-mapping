import numpy as np

from threading import Event
from typing import Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.joysticks import FutabaT6K
from smg.relocalisation.poseglobalisers import HeightBasedMonocularPoseGlobaliser
from smg.rotory.drones import Drone
from smg.utility import ImageUtil, SequenceUtil


class HeightBasedMetricDroneFSM:
    """A finite state machine that allows metric tracking to be configured for a drone by using the drone's height."""

    # NESTED TYPES

    class EDroneState(int):
        """The different states in which a drone can be."""
        pass

    # Fly around as normal with non-metric tracking.
    DS_NON_METRIC = EDroneState(0)
    # Train the globaliser to estimate the scale.
    DS_TRAINING = EDroneState(1)
    # Fly around as normal with metric tracking.
    DS_METRIC = EDroneState(2)

    # CONSTRUCTOR

    def __init__(self, drone: Drone, joystick: FutabaT6K, mapping_client: Optional[MappingClient] = None, *,
                 output_dir: Optional[str] = None, save_frames: bool = False):
        """
        Construct a finite state machine that allows metric tracking to be configured for a drone by using
        the drone's height.

        :param drone:           The drone.
        :param joystick:        The joystick that will be used to control the drone's movement.
        :param mapping_client:  The mapping client to use (if any).
        :param output_dir:      An optional directory into which to save output files.
        :param save_frames:     Whether to save the sequence of frames that have been obtained from the drone.
        """
        self.__calibration_message_sent: bool = False
        self.__drone: Drone = drone
        self.__frame_idx: int = 0
        self.__joystick: FutabaT6K = joystick
        self.__landing_event: Event = Event()
        self.__mapping_client: Optional[MappingClient] = mapping_client
        self.__output_dir: Optional[str] = output_dir
        self.__pose_globaliser: HeightBasedMonocularPoseGlobaliser = HeightBasedMonocularPoseGlobaliser(debug=True)
        self.__save_frames: bool = save_frames
        self.__should_terminate: bool = False
        self.__state: HeightBasedMetricDroneFSM.EDroneState = HeightBasedMetricDroneFSM.DS_NON_METRIC
        self.__takeoff_event: Event = Event()
        self.__throttle_down_event: Event = Event()
        self.__throttle_prev: Optional[float] = None
        self.__throttle_up_event: Event = Event()
        self.__tracker_w_t_c: Optional[np.ndarray] = None

    # PUBLIC METHODS

    def alive(self) -> bool:
        """
        Get whether or not the state machine is still alive.

        :return:    True, if the state machine is still alive, or False otherwise.
        """
        return not self.__should_terminate

    def get_state(self) -> EDroneState:
        """
        Get the state of the drone.

        :return:    The state of the drone.
        """
        return self.__state

    def get_tracker_w_t_c(self) -> Optional[np.ndarray]:
        """
        Try to get a metric transformation from current camera space to world space, as estimated by the tracker.

        .. note::
            This returns None iff either (i) the tracker failed, or (ii) metric tracking hasn't been configured yet.

        :return:    A metric transformation from current camera space to world space, as estimated by the tracker,
                    if available, or None otherwise.
        """
        return self.__tracker_w_t_c

    def iterate(self, image: np.ndarray, image_timestamp: Optional[float],
                intrinsics: Tuple[float, float, float, float],
                tracker_c_t_i: Optional[np.ndarray], height: float,
                takeoff_requested: bool, landing_requested: bool) -> None:
        """
        Run an iteration of the state machine.

        :param image:               The most recent image from the drone.
        :param image_timestamp:     The timestamp of the most recent image from the drone (if known).
        :param intrinsics:          The intrinsics of the drone's camera.
        :param tracker_c_t_i:       A non-metric transformation from initial camera space to current camera space,
                                    as estimated by the tracker.
        :param height:              The most recent height (in m) for the drone.
        :param takeoff_requested:   Whether or not the user has asked for the drone to take off.
        :param landing_requested:   Whether or not the user has asked for the drone to land.
        """
        # If we're connected to a mapping server, send across the camera parameters if we haven't already.
        if self.__mapping_client is not None and not self.__calibration_message_sent:
            height, width = image.shape[:2]
            self.__mapping_client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                (width, height), (width, height), intrinsics, intrinsics
            ))
            self.__calibration_message_sent = True

        # Process any take-off or landing requests, and set the corresponding events so that individual states
        # can respond to them later if desired.
        if takeoff_requested:
            self.__drone.takeoff()
            self.__takeoff_event.set()
        elif landing_requested:
            self.__drone.land()
            self.__landing_event.set()

        # Check for any throttle up/down events that have occurred so that individual states can respond to them later.
        throttle: float = self.__joystick.get_throttle()
        if self.__throttle_prev is not None:
            if throttle <= 0.25 < self.__throttle_prev:
                self.__throttle_down_event.set()
            if throttle >= 0.75 > self.__throttle_prev:
                self.__throttle_up_event.set()

        # Update the drone's movement based on the pitch, roll and yaw values output by the joystick.
        self.__drone.move_forward(self.__joystick.get_pitch())
        self.__drone.turn(self.__joystick.get_yaw())

        if self.__joystick.get_button(1) == 0:
            self.__drone.move_right(0)
            self.__drone.move_up(self.__joystick.get_roll())
        else:
            self.__drone.move_right(self.__joystick.get_roll())
            self.__drone.move_up(0)

        # If the non-metric tracker pose is available, compute its inverse for later use.
        tracker_i_t_c: Optional[np.ndarray] = np.linalg.inv(tracker_c_t_i) if tracker_c_t_i is not None else None

        # Run an iteration of the current state.
        if self.__state == HeightBasedMetricDroneFSM.DS_NON_METRIC:
            self.__iterate_non_metric()
        elif self.__state == HeightBasedMetricDroneFSM.DS_TRAINING:
            self.__iterate_training(tracker_i_t_c, height)
        elif self.__state == HeightBasedMetricDroneFSM.DS_METRIC:
            self.__iterate_metric(image, image_timestamp, intrinsics, tracker_i_t_c, height)

        # Record the current setting of the throttle for later, so we can detect throttle up/down events that occur.
        self.__throttle_prev = throttle

        # Clear any events that have occurred during this iteration of the state machine.
        self.__landing_event.clear()
        self.__takeoff_event.clear()
        self.__throttle_down_event.clear()
        self.__throttle_up_event.clear()

    def terminate(self) -> None:
        """Tell the state machine to terminate."""
        if self.alive():
            if self.__mapping_client is not None:
                self.__mapping_client.terminate()

            self.__should_terminate = True

    # PRIVATE METHODS

    def __iterate_metric(self, image: np.ndarray, image_timestamp: Optional[float],
                         intrinsics: Tuple[float, float, float, float],
                         tracker_i_t_c: Optional[np.ndarray], height: Optional[float]) -> None:
        """
        Run an iteration of the 'metric' state.

        .. note::
            The drone enters this state by throttling down after training the globaliser. It then never leaves
            this state. On entering this state, the throttle will be down.

        :param image:           The most recent image from the drone.
        :param image_timestamp: The timestamp of the most recent image from the drone (if known).
        :param intrinsics:      The drone's camera intrinsics.
        :param tracker_i_t_c:   A non-metric transformation from current camera space to initial camera space,
                                as estimated by the tracker.
        :param height:          The most recent height (in m) for the drone.
        """
        # If the non-metric tracker pose is available:
        if tracker_i_t_c is not None:
            # Use the globaliser to obtain the metric tracker pose.
            self.__tracker_w_t_c = self.__pose_globaliser.apply(tracker_i_t_c)

            # Make a dummy depth image.
            dummy_depth_image: np.ndarray = np.zeros(image.shape[:2], dtype=np.float32)

            # If we're reconstructing a map:
            if self.__mapping_client is not None:
                # Send the current frame across to the mapping server.
                self.__mapping_client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                    self.__frame_idx, image, ImageUtil.to_short_depth(dummy_depth_image), self.__tracker_w_t_c, msg,
                    frame_timestamp=image_timestamp
                ))

            # If an output directory was specified and we're saving frames, save the frame to disk.
            if self.__output_dir is not None and self.__save_frames:
                SequenceUtil.save_rgbd_frame(
                    self.__frame_idx, self.__output_dir, image, dummy_depth_image, self.__tracker_w_t_c,
                    colour_intrinsics=intrinsics, depth_intrinsics=intrinsics
                )

            # Increment the frame index.
            self.__frame_idx += 1
        else:
            # If the non-metric tracker pose isn't available, the metric tracker pose clearly can't be estimated.
            self.__tracker_w_t_c = None

        # Print the tracker pose.
        print("Tracker Pose:")
        print(self.__tracker_w_t_c)

        # If the height is available, also print that.
        if height is not None:
            print(f"Height: {height}")

    def __iterate_non_metric(self) -> None:
        """
        Run an iteration of the 'non-metric' state.

        .. note::
            The drone starts in this state. It leaves this state by throttling up to enter the training state.
            Whilst in this state, the drone can move around as normal, but any poses estimated by the tracker
            will be non-metric.
        """
        # If the user throttles up, start the calibration process.
        if self.__throttle_up_event.is_set():
            self.__state = HeightBasedMetricDroneFSM.DS_TRAINING

    def __iterate_training(self, tracker_i_t_c: Optional[np.ndarray], height: Optional[float]) -> None:
        """
        Run an iteration of the 'training' state.

        .. note::
            The drone enters this state by throttling up from the non-metric state. It leaves this state by
            throttling down again to enter the metric state. On entering this state, the throttle will be up.
            Whilst in this state, the drone should only move up or down, or the scale estimation will fail.

        :param tracker_i_t_c:   A non-metric transformation from current camera space to initial camera space,
                                as estimated by the tracker.
        :param height:          The most recent height (in m) for the drone.
        """
        # Train the pose globaliser if possible.
        if tracker_i_t_c is not None and height is not None:
            self.__pose_globaliser.train(tracker_i_t_c, height)

        # If the user throttles down, complete the calibration process.
        if self.__throttle_down_event.is_set():
            self.__state = HeightBasedMetricDroneFSM.DS_METRIC
