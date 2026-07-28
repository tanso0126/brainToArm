// arm_controller.ino
// Serial-driven angle control for the 6-servo robot arm.
//
// Replaces the hardcoded demo (old example.cpp). The laptop is the brain:
// it streams target joint angles over USB serial; this firmware just moves
// the servos there smoothly and reports back. No intelligence lives here.
//
// Motor map (per user's build), bottom -> top:
//   servo1 (pin 13) : base yaw   (rotate whole arm about Z)
//   servo2 (pin 12) : shoulder   (1st bend)
//   servo3 (pin 11) : elbow      (2nd bend)
//   servo4 (pin 10) : wrist pitch (verified safe range 130..180)
//   servo5 (pin  9) : gripper    (90=open, 170=closed)
//   servo6 (pin  8) : wrist roll
//
// Serial protocol (115200 baud, newline-terminated ASCII):
//   Command from laptop:
//     "A a1 a2 a3 a4 a5 a6\n"      set target angles (deg 0..180) for servos 1..6
//                                   use -1 to leave a joint's target unchanged
//     "P\n"                        ping -> replies "PONG"
//     "S\n"                        status -> replies current angles "C a1..a6"
//     "H\n"                        compiled home pose -> "H a1..a6"
//     "F\n"                        grip sensor -> "F 0..1023", or "F -1" absent
//     "D\n"                        ultrasonic range -> "D millimetres", or
//                                   "D -1" on timeout/out-of-range
//   Reply to laptop:
//     "OK\n"      command accepted
//     "ERR ...\n" parse error
//     when all joints have reached their targets: "DONE\n" (sent once)
//
// Smoothing: each joint slews toward its target at SLEW_DEG_PER_TICK so the
// arm never snaps. The laptop can send a new target at any time.

#include <Servo.h>
#include "home_pose.h"

const uint8_t N = 6;
const uint8_t PINS[N] = {13, 12, 11, 10, 9, 8};
const uint8_t ULTRASONIC_TRIGGER_PIN = 7;
const uint8_t ULTRASONIC_ECHO_PIN = 6;
const unsigned long ULTRASONIC_TIMEOUT_US = 25000UL;
const unsigned int ULTRASONIC_MIN_MM = 20;
const unsigned int ULTRASONIC_MAX_MM = 4000;

// Physically verified travel limits after removing the old unused third motor.
const int MIN_DEG[N] = {0, 0, 0, 130, 90, 0};
const int MAX_DEG[N] = {180, 150, 180, 180, 180, 180};

// Shared with laptop/config.py through home_pose.h; edit the six named values
// there rather than duplicating the startup pose in multiple files.
const int HOME_DEG[N] = ARM_HOME_VALUES;

const float SLEW_DEG_PER_TICK = 1.5; // max move per control tick (~ speed limit)
const int   TICK_MS = 15;            // control loop period
const int   REACH_EPS = 1;           // within this many deg counts as "reached"

// Optional physical grasp feedback. Install an FSR voltage divider or analog
// current-sensor output on A0, then set true and calibrate the host threshold.
// A plain PWM hobby servo does not report torque or its actual shaft position.
const bool GRIP_FEEDBACK_ENABLED = false;
const uint8_t GRIP_FEEDBACK_PIN = A0;

Servo servos[N];
float current[N];   // live angle (float for smooth slew)
int   target[N];    // commanded angle
int   lastWritten[N]; // avoid rewriting an unchanged pulse command every tick
bool  announced;    // have we already sent DONE for the current target set?

int clampJoint(uint8_t i, int deg) {
  if (deg < MIN_DEG[i]) return MIN_DEG[i];
  if (deg > MAX_DEG[i]) return MAX_DEG[i];
  return deg;
}

void setup() {
  Serial.begin(115200);
  pinMode(ULTRASONIC_TRIGGER_PIN, OUTPUT);
  digitalWrite(ULTRASONIC_TRIGGER_PIN, LOW);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);
  for (uint8_t i = 0; i < N; i++) {
    servos[i].attach(PINS[i]);
    current[i] = HOME_DEG[i];
    target[i]  = HOME_DEG[i];
    servos[i].write((int)current[i]);
    lastWritten[i] = (int)current[i];
  }
  announced = true;
  Serial.println("READY");
}

bool parseIntStrict(char* tok, int* out) {
  if (tok == NULL || *tok == '\0') return false;
  char* end = NULL;
  long value = strtol(tok, &end, 10);
  if (end == tok || *end != '\0' || value < -1 || value > 180) return false;
  *out = (int)value;
  return true;
}

// Parse "A a1 a2 ... a6" atomically. -1 keeps the existing target. A malformed
// command changes no targets; this matters when a truncated serial line arrives.
bool parseAngles(char* line) {
  if (line[1] != ' ' && line[1] != '\t') return false;
  int nextTarget[N];
  for (uint8_t i = 0; i < N; i++) nextTarget[i] = target[i];
  char* tok = strtok(line + 1, " ,\t"); // skip the 'A'
  for (uint8_t i = 0; i < N; i++) {
    int v;
    if (!parseIntStrict(tok, &v)) return false;
    if (v >= 0) nextTarget[i] = clampJoint(i, v);
    tok = strtok(NULL, " ,\t");
  }
  if (tok != NULL) return false;
  for (uint8_t i = 0; i < N; i++) target[i] = nextTarget[i];
  return true;
}

void sendStatus() {
  Serial.print("C");
  for (uint8_t i = 0; i < N; i++) {
    Serial.print(' ');
    Serial.print((int)(current[i] + 0.5));
  }
  Serial.println();
}

void sendHomePose() {
  Serial.print("H");
  for (uint8_t i = 0; i < N; i++) {
    Serial.print(' ');
    Serial.print(HOME_DEG[i]);
  }
  Serial.println();
}

void sendGripFeedback() {
  if (!GRIP_FEEDBACK_ENABLED) {
    Serial.println("F -1");
    return;
  }
  long sum = 0;
  for (uint8_t i = 0; i < 8; i++) sum += analogRead(GRIP_FEEDBACK_PIN);
  Serial.print("F ");
  Serial.println((int)(sum / 8));
}

void sendUltrasonicDistance() {
  // Trigger only on an explicit host request. Continuous pulseIn() calls would
  // unnecessarily stall the servo slew loop whenever no echo is received.
  digitalWrite(ULTRASONIC_TRIGGER_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIGGER_PIN, LOW);

  unsigned long durationUs = pulseIn(
    ULTRASONIC_ECHO_PIN, HIGH, ULTRASONIC_TIMEOUT_US);
  if (durationUs == 0) {
    Serial.println("D -1");
    return;
  }

  // 343 m/s = 0.343 mm/us; divide by two for the outbound/return path.
  unsigned long distanceMm = (durationUs * 343UL + 1000UL) / 2000UL;
  if (distanceMm < ULTRASONIC_MIN_MM || distanceMm > ULTRASONIC_MAX_MM) {
    Serial.println("D -1");
    return;
  }
  Serial.print("D ");
  Serial.println(distanceMm);
}

void handleLine(char* line) {
  switch (line[0]) {
    case 'A':
      if (parseAngles(line)) { announced = false; Serial.println("OK"); }
      else                    Serial.println("ERR parse");
      break;
    case 'P':
      if (line[1] == '\0') Serial.println("PONG");
      else                  Serial.println("ERR parse");
      break;
    case 'S':
      if (line[1] == '\0') sendStatus();
      else                  Serial.println("ERR parse");
      break;
    case 'H':
      if (line[1] == '\0') sendHomePose();
      else                  Serial.println("ERR parse");
      break;
    case 'F':
      if (line[1] == '\0') sendGripFeedback();
      else                  Serial.println("ERR parse");
      break;
    case 'D':
      if (line[1] == '\0') sendUltrasonicDistance();
      else                  Serial.println("ERR parse");
      break;
    default:  Serial.println("ERR cmd");
  }
}

void readSerial() {
  static char buf[64];
  static uint8_t len = 0;
  static bool overflow = false;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (overflow) {
        Serial.println("ERR line too long");
      } else if (len > 0) {
        buf[len] = '\0';
        handleLine(buf);
      }
      len = 0;
      overflow = false;
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = c;
    } else {
      overflow = true;
    }
  }
}

void slew() {
  bool allReached = true;
  for (uint8_t i = 0; i < N; i++) {
    float diff = target[i] - current[i];
    if (fabs(diff) <= REACH_EPS) {
      current[i] = target[i];
    } else {
      current[i] += (diff > 0 ? 1 : -1) * min(SLEW_DEG_PER_TICK, (float)fabs(diff));
      allReached = false;
    }
    int output = (int)(current[i] + 0.5);
    if (output != lastWritten[i]) {
      servos[i].write(output);
      lastWritten[i] = output;
    }
  }
  if (allReached && !announced) { Serial.println("DONE"); announced = true; }
}

void loop() {
  readSerial();
  slew();
  delay(TICK_MS);
}
