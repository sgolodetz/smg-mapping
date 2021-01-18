import numpy as np

from threading import Event
from typing import Optional, Tuple

from smg.relocalisation.poseglobalisers import MonocularPoseGlobaliser
from smg.rotory.drones import Drone
from smg.rotory.joysticks import FutabaT6K
from smg.utility import ImageUtil

from ..remote import MappingClient, RGBDFrameMessageUtil


class EDroneCalibrationState(int):
    """The different calibration states in which a drone can be."""
    pass


# Fly around as normal with non-metric tracking.
DCS_UNCALIBRATED: EDroneCalibrationState = 0
# Fly around in front of the marker to set the reference space.
DCS_SETTING_REFERENCE: EDroneCalibrationState = 1
# Land prior to training the globaliser to estimate the scale.
DCS_PREPARING_TO_TRAIN: EDroneCalibrationState = 2
# Whilst on the ground, train the globaliser to estimate the scale.
DCS_TRAINING: EDroneCalibrationState = 3
# Fly around as normal with metric tracking.
DCS_CALIBRATED: EDroneCalibrationState = 4


class DroneFSM:
    """A finite state machine for a drone."""

    # CONSTRUCTOR

    def __init__(self, drone: Drone, joystick: FutabaT6K, *, reconstruct: bool):
        """
        Construct a finite state machine for a drone.

        :param drone:       The drone.
        :param joystick:    The joystick that will be used to control the drone's movement.
        :param reconstruct: Whether to connect to the mapping server to reconstruct a map.
        """
        self.__calibration_state: EDroneCalibrationState = DCS_UNCALIBRATED
        self.__drone: Drone = drone
        self.__joystick: FutabaT6K = joystick
        self.__landing_event: Event = Event()
        self.__pose_globaliser: MonocularPoseGlobaliser = MonocularPoseGlobaliser(debug=True)
        self.__relocaliser_w_t_c_for_training: Optional[np.ndarray] = None
        self.__takeoff_event: Event = Event()
        self.__throttle_down_event: Event = Event()
        self.__throttle_prev: Optional[float] = None
        self.__throttle_up_event: Event = Event()
        self.__tracker_w_t_c: Optional[np.ndarray] = None
        self.__should_terminate: bool = False

        self.__calibration_message_sent: bool = False
        self.__frame_idx: int = 0
        self.__mapping_client: Optional[MappingClient] = None
        if reconstruct:
            self.__mapping_client = MappingClient(
                frame_compressor=RGBDFrameMessageUtil.compress_frame_message
            )

    # PUBLIC METHODS

    def alive(self) -> bool:
        """
        Get whether or not the state machine is still alive.

        :return:    True, if the state machine is still alive, or False otherwise.
        """
        return not self.__should_terminate

    def get_calibration_state(self) -> EDroneCalibrationState:
        """
        Get the calibration state of the drone.

        :return:    The calibration state of the drone.
        """
        return self.__calibration_state

    def get_tracker_w_t_c(self) -> Optional[np.ndarray]:
        """
        Try to get a metric transformation from current camera space to world space, as estimated by the tracker.

        .. note::
            This returns None iff either (i) the tracker failed, or (ii) the drone hasn't been calibrated yet.

        :return:    A metric transformation from current camera space to world space, as estimated by the tracker,
                    if available, or None otherwise.
        """
        return self.__tracker_w_t_c

    def iterate(self, image: np.ndarray, intrinsics: Tuple[float, float, float, float],
                tracker_c_t_i: Optional[np.ndarray], relocaliser_w_t_c: Optional[np.ndarray],
                takeoff_requested: bool, landing_requested: bool) -> None:
        """
        Run an iteration of the state machine.

        :param image:               The most recent image from the drone.
        :param intrinsics:          The intrinsics of the drone's camera.
        :param tracker_c_t_i:       A non-metric transformation from initial camera space to current camera space,
                                    as estimated by the tracker.
        :param relocaliser_w_t_c:   A metric transformation from current camera space to world space, as estimated
                                    by the relocaliser.
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
            # self.__drone.takeoff()
            self.__takeoff_event.set()
        elif landing_requested:
            # self.__drone.land()
            self.__landing_event.set()

        # Check for any throttle up/down events that have occurred so that individual states can respond to them later.
        throttle: float = self.__joystick.get_throttle()
        if self.__throttle_prev is not None:
            if throttle <= 0.25 < self.__throttle_prev:
                self.__throttle_down_event.set()
            if throttle >= 0.75 > self.__throttle_prev:
                self.__throttle_up_event.set()

        # Update the drone's movement based on the pitch, roll and yaw values output by the joystick.
        # self.__drone.move_forward(self.__joystick.get_pitch())
        # self.__drone.turn(self.__joystick.get_yaw())
        #
        # if self.__joystick.get_button(1) == 0:
        #     self.__drone.move_right(0)
        #     self.__drone.move_up(self.__joystick.get_roll())
        # else:
        #     self.__drone.move_right(self.__joystick.get_roll())
        #     self.__drone.move_up(0)

        # If the non-metric tracker pose is available, compute its inverse for later use.
        tracker_i_t_c: Optional[np.ndarray] = np.linalg.inv(tracker_c_t_i) if tracker_c_t_i is not None else None

        # Run an iteration of the current state.
        if self.__calibration_state == DCS_UNCALIBRATED:
            self.__iterate_uncalibrated()
        elif self.__calibration_state == DCS_SETTING_REFERENCE:
            self.__iterate_setting_reference(tracker_i_t_c, relocaliser_w_t_c)
        elif self.__calibration_state == DCS_PREPARING_TO_TRAIN:
            self.__iterate_preparing_to_train()
        elif self.__calibration_state == DCS_TRAINING:
            self.__iterate_training(tracker_i_t_c)
        elif self.__calibration_state == DCS_CALIBRATED:
            self.__iterate_calibrated(image, tracker_i_t_c, relocaliser_w_t_c)

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

    def __iterate_calibrated(self, image: np.ndarray, tracker_i_t_c: Optional[np.ndarray],
                             relocaliser_w_t_c: Optional[np.ndarray]) -> None:
        """
        Run an iteration of the 'calibrated' state.

        .. note::
            The drone enters this state by taking off after training the globaliser. It then never leaves this state.
            On entering this state, the throttle will be down (as it was during the training of the globaliser).
            Moving the throttle up/down will then set/clear a fixed height.

        :param image:               The most recent image from the drone.
        :param tracker_i_t_c:       A non-metric transformation from current camera space to initial camera space,
                                    as estimated by the tracker.
        :param relocaliser_w_t_c:   A metric transformation from current camera space to world space, as estimated
                                    by the relocaliser.
        """
        # If the user throttles down, clear the fixed height.
        if self.__throttle_down_event.is_set():
            self.__pose_globaliser.clear_fixed_height()

        # If the non-metric tracker pose is available:
        if tracker_i_t_c is not None:
            # Use the globaliser to obtain the metric tracker pose.
            self.__tracker_w_t_c = self.__pose_globaliser.apply(tracker_i_t_c)

            # If the user throttles up, set the current height as the fixed height. Note that it is theoretically
            # possible for the user to throttle up during a period of tracking failure. In that case, the throttle
            # will be up but no fixed height will have been set. However, if that happens, the user can simply
            # throttle down again with no ill effects (clearing a fixed height that hasn't been set is a no-op).
            if self.__throttle_up_event.is_set():
                self.__pose_globaliser.set_fixed_height(self.__tracker_w_t_c)

            # If we're reconstructing a map, send the current frame across to the mapping server.
            if self.__mapping_client is not None:
                dummy_depth_image: np.ndarray = np.zeros(image.shape[:2], dtype=np.float32)
                self.__mapping_client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                    self.__frame_idx, image, ImageUtil.to_short_depth(dummy_depth_image), self.__tracker_w_t_c, msg
                ))
                self.__frame_idx += 1
        else:
            # If the non-metric tracker pose isn't available, the metric tracker pose clearly can't be estimated.
            self.__tracker_w_t_c = None

        # Print the tracker pose.
        print("Tracker Pose:")
        print(self.__tracker_w_t_c)

        # If the relocaliser pose is available, also print that.
        if relocaliser_w_t_c is not None:
            print("Relocaliser Pose:")
            print(relocaliser_w_t_c)

    def __iterate_preparing_to_train(self) -> None:
        """
        Run an iteration of the 'preparing to train' state.

        .. note::
            The drone enters this state either by landing after setting the globaliser's reference space,
            or by throttling up after training the globaliser. It leaves this state either by throttling
            down to enter the training state, or by taking off to re-enter the setting reference state.
            On entering this state, the throttle will be up.
        .. note::
            In practice, this state exists to allow the drone to land prior to starting to train the
            globaliser. The training process should only be started once the drone is on the ground.
        """
        # If the user has told the drone to take off, return to the previous calibration step.
        if self.__takeoff_event.is_set():
            self.__calibration_state = DCS_SETTING_REFERENCE

        # If the user has throttled down, move on to the next calibration step.
        if self.__throttle_down_event.is_set():
            self.__calibration_state = DCS_TRAINING

    def __iterate_setting_reference(self, tracker_i_t_c: Optional[np.ndarray],
                                    relocaliser_w_t_c: Optional[np.ndarray]) -> None:
        """
        Run an iteration of the 'setting reference' state.

        .. note::
            The drone enters this state either by throttling up from the uncalibrated state, or by taking off
            again after preparing to train the globaliser. It leaves this state either by starting to land and
            entering the preparing to train state, or by throttling down and re-entering the uncalibrated state.
            On entering this state, the throttle will be up. To fulfil the objective of being in this state,
            the drone must at some point fly in front of the marker so that the reference space can be set.
            It can only land and continue with calibration from a point at which it can see the marker (to
            ensure that the reference space has been set correctly).

        :param tracker_i_t_c:       A non-metric transformation from current camera space to initial camera space,
                                    as estimated by the tracker.
        :param relocaliser_w_t_c:   A metric transformation from current camera space to world space, as estimated
                                    by the relocaliser.
        """
        # If the drone's successfully relocalised using the marker:
        if relocaliser_w_t_c is not None:
            # Set the pose globaliser's reference space to get it ready for training. Note that this can safely be
            # called repeatedly (the poses from the most recent call will be used to define the reference space).
            self.__pose_globaliser.set_reference_space(tracker_i_t_c, relocaliser_w_t_c)

            # If the user has told the drone to land, move on to the next calibration step. Otherwise, stay on this
            # step, and wait for the user to take off and try again.
            if self.__landing_event.is_set():
                self.__calibration_state = DCS_PREPARING_TO_TRAIN

                # It's unlikely that we'll be able to see the ArUco marker to relocalise once we're on the ground,
                # so estimate the relocaliser pose we'll have at that point by using the pose currently output by
                # the relocaliser and the fact that we'll be on the ground (i.e. y = 0) then.
                self.__relocaliser_w_t_c_for_training = relocaliser_w_t_c.copy()
                self.__relocaliser_w_t_c_for_training[1, 3] = 0.0

        # If the user has throttled down, stop the calibration process.
        if self.__throttle_down_event.is_set():
            self.__calibration_state = DCS_UNCALIBRATED

    def __iterate_training(self, tracker_i_t_c: Optional[np.ndarray]) -> None:
        """
        Run an iteration of the 'training' state.

        .. note::
            The drone enters this state by throttling down from the preparing to train state. It leaves this
            state either by taking off to enter the calibrated state, or by throttling up again to return to
            the preparing to train state. On entering this state, the throttle will be down. The drone will
            be on the ground whilst in this state.

        :param tracker_i_t_c:   A non-metric transformation from current camera space to initial camera space,
                                as estimated by the tracker.
        """
        # Train the pose globaliser if possible.
        if tracker_i_t_c is not None and self.__relocaliser_w_t_c_for_training is not None:
            self.__pose_globaliser.train(tracker_i_t_c, self.__relocaliser_w_t_c_for_training)

        # If the user has told the drone to take off, complete the calibration process.
        if self.__takeoff_event.is_set():
            self.__calibration_state = DCS_CALIBRATED

        # If the user has throttled up, return to the previous calibration step.
        if self.__throttle_up_event.is_set():
            self.__calibration_state = DCS_PREPARING_TO_TRAIN

    def __iterate_uncalibrated(self) -> None:
        """
        Run an iteration of the 'uncalibrated' state.

        .. note::
            The drone starts in this state. It leaves this state by throttling up to enter the
            setting reference state. Whilst in this state, the drone can move around as normal,
            but any poses estimated by the tracker will be non-metric.
        """
        # If the user throttles up, start the calibration process.
        if self.__throttle_up_event.is_set():
            self.__calibration_state = DCS_SETTING_REFERENCE
