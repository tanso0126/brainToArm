#pragma once

// Reduced physical arm: only servos 2, 3, and 5 are electrically driven.
// These six values are protocol placeholders/status values.  The mechanical
// wrist angle is configured in laptop/reduced_dof.py because servo 4 no longer
// exists and cannot report an angle.
#define REDUCED_HOME_SERVO_1 90
#define REDUCED_HOME_SERVO_2 70
#define REDUCED_HOME_SERVO_3 90
#define REDUCED_HOME_SERVO_4 140
#define REDUCED_HOME_SERVO_5 170
#define REDUCED_HOME_SERVO_6 180

#define REDUCED_HOME_VALUES { \
  REDUCED_HOME_SERVO_1, REDUCED_HOME_SERVO_2, REDUCED_HOME_SERVO_3, \
  REDUCED_HOME_SERVO_4, REDUCED_HOME_SERVO_5, REDUCED_HOME_SERVO_6 \
}
