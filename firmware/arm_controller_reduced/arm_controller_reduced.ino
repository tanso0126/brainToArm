// Reduced brainToArm firmware.
//
// Active outputs only:
//   logical servo 2 -> D12 shoulder
//   logical servo 3 -> D11 elbow
//   logical servo 5 -> D9  gripper
//
// D13 (base), D10 (old wrist pitch), and D8 (old wrist roll) are NEVER
// attached, so the firmware sends no PWM to the failed/unused motors.  The
// six-field serial protocol is retained so the old firmware and old host code
// remain untouched in their original directories.

#include <Servo.h>
#include "reduced_pose.h"

const uint8_t N = 6;
const uint8_t ACTIVE_COUNT = 3;
const uint8_t ACTIVE_LOGICAL[ACTIVE_COUNT] = {1, 2, 4}; // zero-based 2,3,5
const uint8_t ACTIVE_PINS[ACTIVE_COUNT] = {12, 11, 9};
const uint8_t ULTRASONIC_TRIGGER_PIN = 7;
const uint8_t ULTRASONIC_ECHO_PIN = 6;
const unsigned long ULTRASONIC_TIMEOUT_US = 25000UL;
const unsigned int ULTRASONIC_MIN_MM = 20;
const unsigned int ULTRASONIC_MAX_MM = 4000;

const int MIN_DEG[N] = {90, 65, 35, 140, 90, 180};
const int MAX_DEG[N] = {90, 145, 165, 140, 180, 180};
const int HOME_DEG[N] = REDUCED_HOME_VALUES;
const float SLEW_DEG_PER_TICK = 1.5;
const int TICK_MS = 15;
const int REACH_EPS = 1;

Servo activeServos[ACTIVE_COUNT];
float current[N];
int target[N];
int lastWritten[ACTIVE_COUNT];
bool announced;

bool isActiveLogical(uint8_t logical) {
  for (uint8_t slot = 0; slot < ACTIVE_COUNT; slot++) {
    if (ACTIVE_LOGICAL[slot] == logical) return true;
  }
  return false;
}

int activeSlot(uint8_t logical) {
  for (uint8_t slot = 0; slot < ACTIVE_COUNT; slot++) {
    if (ACTIVE_LOGICAL[slot] == logical) return slot;
  }
  return -1;
}

int clampJoint(uint8_t logical, int deg) {
  if (deg < MIN_DEG[logical]) return MIN_DEG[logical];
  if (deg > MAX_DEG[logical]) return MAX_DEG[logical];
  return deg;
}

void setup() {
  Serial.begin(115200);
  pinMode(ULTRASONIC_TRIGGER_PIN, OUTPUT);
  digitalWrite(ULTRASONIC_TRIGGER_PIN, LOW);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);
  for (uint8_t logical = 0; logical < N; logical++) {
    current[logical] = HOME_DEG[logical];
    target[logical] = HOME_DEG[logical];
  }
  for (uint8_t slot = 0; slot < ACTIVE_COUNT; slot++) {
    uint8_t logical = ACTIVE_LOGICAL[slot];
    activeServos[slot].attach(ACTIVE_PINS[slot]);
    activeServos[slot].write(HOME_DEG[logical]);
    lastWritten[slot] = HOME_DEG[logical];
  }
  announced = true;
  Serial.println("READY REDUCED_2DOF");
}

bool parseIntStrict(char* token, int* out) {
  if (token == NULL || *token == '\0') return false;
  char* end = NULL;
  long value = strtol(token, &end, 10);
  if (end == token || *end != '\0' || value < -1 || value > 180) return false;
  *out = (int)value;
  return true;
}

bool parseAngles(char* line) {
  if (line[1] != ' ' && line[1] != '\t') return false;
  int nextTarget[N];
  for (uint8_t i = 0; i < N; i++) nextTarget[i] = target[i];
  char* token = strtok(line + 1, " ,\t");
  for (uint8_t logical = 0; logical < N; logical++) {
    int value;
    if (!parseIntStrict(token, &value)) return false;
    if (value >= 0) {
      if (!isActiveLogical(logical) && value != HOME_DEG[logical]) {
        return false; // a host tried to command disabled 1/4/6
      }
      nextTarget[logical] = clampJoint(logical, value);
    }
    token = strtok(NULL, " ,\t");
  }
  if (token != NULL) return false;
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

void sendUltrasonicDistance() {
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
      else Serial.println("ERR reduced fixed joint or parse");
      break;
    case 'P':
      if (line[1] == '\0') Serial.println("PONG");
      else Serial.println("ERR parse");
      break;
    case 'S':
      if (line[1] == '\0') sendStatus();
      else Serial.println("ERR parse");
      break;
    case 'H':
      if (line[1] == '\0') sendHomePose();
      else Serial.println("ERR parse");
      break;
    case 'F':
      if (line[1] == '\0') Serial.println("F -1");
      else Serial.println("ERR parse");
      break;
    case 'D':
      if (line[1] == '\0') sendUltrasonicDistance();
      else Serial.println("ERR parse");
      break;
    default:
      Serial.println("ERR cmd");
  }
}

void readSerial() {
  static char buffer[64];
  static uint8_t length = 0;
  static bool overflow = false;
  while (Serial.available()) {
    char value = Serial.read();
    if (value == '\n' || value == '\r') {
      if (overflow) Serial.println("ERR line too long");
      else if (length > 0) {
        buffer[length] = '\0';
        handleLine(buffer);
      }
      length = 0;
      overflow = false;
    } else if (length < sizeof(buffer) - 1) {
      buffer[length++] = value;
    } else {
      overflow = true;
    }
  }
}

void slew() {
  bool allReached = true;
  for (uint8_t slot = 0; slot < ACTIVE_COUNT; slot++) {
    uint8_t logical = ACTIVE_LOGICAL[slot];
    float difference = target[logical] - current[logical];
    if (fabs(difference) <= REACH_EPS) {
      current[logical] = target[logical];
    } else {
      current[logical] += (difference > 0 ? 1 : -1) *
        min(SLEW_DEG_PER_TICK, (float)fabs(difference));
      allReached = false;
    }
    int output = (int)(current[logical] + 0.5);
    if (output != lastWritten[slot]) {
      activeServos[slot].write(output);
      lastWritten[slot] = output;
    }
  }
  if (allReached && !announced) {
    Serial.println("DONE");
    announced = true;
  }
}

void loop() {
  readSerial();
  slew();
  delay(TICK_MS);
}
