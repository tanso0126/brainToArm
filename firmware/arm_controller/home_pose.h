#pragma once

// Single source of truth for the seven-servo startup/home pose.
// Change only the number beside the servo you want, then upload the firmware.
#define ARM_HOME_SERVO_1 90
#define ARM_HOME_SERVO_2 90
#define ARM_HOME_SERVO_3 90
#define ARM_HOME_SERVO_4 180
#define ARM_HOME_SERVO_5 90
#define ARM_HOME_SERVO_6 0
#define ARM_HOME_SERVO_7 180

#define ARM_HOME_VALUES { \
  ARM_HOME_SERVO_1, ARM_HOME_SERVO_2, ARM_HOME_SERVO_3, \
  ARM_HOME_SERVO_4, ARM_HOME_SERVO_5, ARM_HOME_SERVO_6, \
  ARM_HOME_SERVO_7 \
}
