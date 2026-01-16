# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import logging
import math
import time
import traceback
from copy import copy, deepcopy

import numpy as np
import tqdm

from lerobot.common.robot_devices.motors.configs import XhandBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

# TODO: 
# 1. resolution and bounds
# 2. datatype
# 3. increment or 

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]

# FILL
LOWER_BOUND_DEGREE = -270
UPPER_BOUND_DEGREE = 270

LOWER_BOUND_LINEAR = -2000
UPPER_BOUND_LINEAR = 2000

HALF_TURN_DEGREE = 180


MODEL_RESOLUTION = {
    "xhand": 2 * math.pi,
}

def convert_degrees_to_steps(degrees: float | np.ndarray, models: str | list[str]) -> np.ndarray:
    """This function converts the degree range to the step range for indicating motors rotation.
    It assumes a motor achieves a full rotation by going from -180 degree position to +180.
    The motor resolution (e.g. 4096) corresponds to the number of steps needed to achieve a full rotation.
    """
    resolutions = [MODEL_RESOLUTION[model] for model in models]
    steps = degrees / 180 * np.array(resolutions) / 2
    steps = steps.astype(int)
    return steps



def get_group_sync_key(data_name, motor_names):
    group_key = f"{data_name}_" + "_".join(motor_names)
    return group_key


def get_result_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    rslt_name = f"{fn_name}_{group_key}"
    return rslt_name


def get_queue_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    queue_name = f"{fn_name}_{group_key}"
    return queue_name


def get_log_name(var_name, fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(enum.Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are expressed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class XhandBus:
    def __init__(
        self,
        config: XhandBusConfig,
    ):
        self.port = config.port
        self.motors = config.motors
        self.hand_id = config.id

        self.model_resolution = deepcopy(MODEL_RESOLUTION)

        self.calibration = None
        self.is_connected = False
        self.logs = {}

        self.robot = None
        self.motion_enable = None

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"XhandBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        from xhand_controller import xhand_control

        self.robot = xhand_control.XHandControl()
        
        serial_port = '/dev/ttyUSB0'
        baud_rate = 3000000
        self.robot.open_serial(serial_port,baud_rate,)
        self.robot.set_hand_id(1,self.hand_id)
        self.is_connected = True
        
        self.hand_command = xhand_control.HandCommand_t()
        for i in range(12):
            self.hand_command.finger_command[i].id = i
            self.hand_command.finger_command[i].kp = 100
            self.hand_command.finger_command[i].ki = 0
            self.hand_command.finger_command[i].kd = 0
            self.hand_command.finger_command[i].position = 0
            if i == 0:
                self.hand_command.finger_command[i].position = 1.04
            self.hand_command.finger_command[i].tor_max = 300
            self.hand_command.finger_command[i].mode = 3


    def reconnect(self):
        from xhand_controller import xhand_control

        self.robot = xhand_control.XHandControl()
        self.is_connected = True


    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_hand_mode(self, mode: int):
            for i in range(12):
                self.hand_command.finger_command[i].mode = mode
            self.robot.send_command(self.hand_id, self.hand_command)
            time.sleep(1)

    def reset_hand(self, mode: int):
        for i in range(12):
            self.hand_command.finger_command[i].id = i
            self.hand_command.finger_command[i].kp = 0
            self.hand_command.finger_command[i].ki = 0
            self.hand_command.finger_command[i].kd = 0
            self.hand_command.finger_command[i].position = 0
            self.hand_command.finger_command[i].tor_max = 0
            self.hand_command.finger_command[i].mode = mode
        self.robot.send_command(self.hand_id, self.hand_command)
        time.sleep(1)

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        try:
            values = self.apply_calibration(values, motor_names)
            
        except JointOutOfRangeError as e:
            print(e)
            self.autocorrect_calibration(values, motor_names)
            values = self.apply_calibration(values, motor_names)
        return values

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # relative direction
                if drive_mode:
                    values[i] *= -1

                # Phase differences
                values[i] += homing_offset

                # From original range [-resolution//2, resolution//2] to universal float32 centered degree range [-180, 180]
                values[i] = values[i] / (resolution // 2) * HALF_TURN_DEGREE

                if (values[i] < LOWER_BOUND_DEGREE) or (values[i] > UPPER_BOUND_DEGREE):
                    #print("xarm non-correction result:",values)
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for xarm: {name}. "
                        f"Expected to be in nominal range of [-{HALF_TURN_DEGREE}, {HALF_TURN_DEGREE}] degrees (a full rotation), "
                        f"with a maximum range of [{LOWER_BOUND_DEGREE}, {UPPER_BOUND_DEGREE}] degrees to account for joints that can rotate a bit more, "
                        f"but present value is {values[i]} degree. "
                        "This might be due to a cable connection issue creating an artificial 360 degrees jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

            # Useless, so not changed.
            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to a nominal range [0, 100] %,
                # useful for joints with linear motions like Aloha gripper
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for xarm: {name}. "
                        f"Expected to be in nominal range of [0, 100] % (a full linear translation), "
                        f"with a maximum range of [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}] % to account for some imprecision during calibration, "
                        f"but present value is {values[i]} %. "
                        "This might be due to a cable connection issue creating an artificial jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

        return values

    def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Update direction of rotation of the motor to match between leader and follower.
                # In fact, the motor of the leader for a given joint can be assembled in an
                # opposite direction in term of rotation than the motor of the follower on the same joint.
                if drive_mode:
                    values[i] *= -1

                # Convert from initial range to range [-180, 180] degrees
                calib_val = (values[i] + homing_offset) / (resolution // 2) * HALF_TURN_DEGREE
                in_range = (calib_val > LOWER_BOUND_DEGREE) and (calib_val < UPPER_BOUND_DEGREE)

                # Solve this inequality to find the factor to shift the range into [-180, 180] degrees
                # values[i] = (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE
                # - HALF_TURN_DEGREE <= (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE <= HALF_TURN_DEGREE
                # (- (resolution // 2) - values[i] - homing_offset) / resolution <= factor <= ((resolution // 2) - values[i] - homing_offset) / resolution
                low_factor = (-(resolution // 2) - values[i] - homing_offset) / resolution
                upp_factor = ((resolution // 2) - values[i] - homing_offset) / resolution

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from initial range to range [0, 100] in %
                calib_val = (values[i] - start_pos) / (end_pos - start_pos) * 100
                in_range = (calib_val > LOWER_BOUND_LINEAR) and (calib_val < UPPER_BOUND_LINEAR)

                # Solve this inequality to find the factor to shift the range into [0, 100] %
                # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos + resolution * factor - start_pos - resolution * factor) * 100
                # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100
                # 0 <= (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100 <= 100
                # (start_pos - values[i]) / resolution <= factor <= (end_pos - values[i]) / resolution
                low_factor = (start_pos - values[i]) / resolution
                upp_factor = (end_pos - values[i]) / resolution

            if not in_range:
                # Get first integer between the two bounds
                if low_factor < upp_factor:
                    factor = math.ceil(low_factor)

                    if factor > upp_factor:
                        raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")
                else:
                    factor = math.ceil(upp_factor)

                    if factor > low_factor:
                        raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")

                if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                    out_of_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
                    in_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
                elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                    out_of_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"
                    in_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"

                logging.warning(
                    f"Auto-correct calibration of motor '{name}' by shifting value by {abs(factor)} full turns, "
                    f"from '{out_of_range_str}' to '{in_range_str}'."
                )

                # A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
                self.calibration["homing_offset"][calib_idx] += resolution * factor

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Convert from nominal 0-centered degree range [-180, 180] to
                # 0-centered resolution range (e.g. [-2048, 2048] for resolution=4096)
                values[i] = values[i] / HALF_TURN_DEGREE * (resolution // 2)

                # Subtract the homing offsets to come back to actual motor range of values
                # which can be arbitrary.
                values[i] -= homing_offset

                # Remove drive mode, which is the rotation direction of the motor, to come back to
                # actual motor rotation direction which can be arbitrary.
                if drive_mode:
                    values[i] *= -1

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from nominal lnear range of [0, 100] % to
                # actual motor range of values which can be arbitrary.
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        # values = np.round(values).astype(np.int32)
        return values

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"XhandBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )
            
        if motor_names is None:
            motor_names = self.motor_names

        start_time = time.perf_counter()


        if data_name == "Present_Position":
            _, state = self.robot.read_state(self.hand_id, False)
            # if error_struct.error_code != 0:
            #     print(f"xhand read_state error:{self.parse_error_code(error_struct)}\n")
            #     return
            
            motor_indexes = [int(''.join(filter(str.isdigit, motor_name))) for motor_name in self.motor_names]
            values = [state.finger_state[i].position for i in motor_indexes]

        elif data_name == "Torque_Enable":
            values = self.motion_enable

        values = np.array(values)
        
        #print("xarm original read:",data_name,values)

        # Convert to signed int to use range [-2048, 2048] for our motor positions.
        # if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
        #     values = values.astype(np.int32)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.apply_calibration_autocorrect(values.copy(), motor_names)
            
        #print("xarm calibrated read:",data_name,values)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"XhandBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        if motor_names is None:
            motor_names = self.motor_names

        start_time = time.perf_counter()
        
        #print("xarm original read:",data_name,values)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.revert_calibration(values.copy(), motor_names)

        #print("xarm reverted read:",data_name,values)

        if data_name == "Torque_Enable":
            # TODO
            mode = 0 if not values else 3
            self.set_hand_mode(mode)
            self.motion_enable = values

        elif data_name == "Goal_Position":
            for i in range(len(values)):
                value = values[i]
                motor_index = int(''.join(filter(str.isdigit, self.motor_names[i])))
                self.hand_command.finger_command[motor_index].position = value
            self.robot.send_command(self.hand_id, self.hand_command)

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"XhandBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        self.reset_hand(mode=0)

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
