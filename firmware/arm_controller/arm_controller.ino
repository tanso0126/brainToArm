// arm_controller.ino
// Serial-driven angle control for the 7-servo robot arm.
//
// Replaces the hardcoded demo (old example.cpp). The laptop is the brain:
// it streams target joint angles over USB serial; this firmware just moves
// the servos there smoothly and reports back. No intelligence lives here.
//
// Motor map (per user's build), bottom -> top:
//   servo1 (pin 13) : base yaw   (rotate whole arm about Z)
//   servo2 (pin 12) : shoulder   (1st bend)
//   servo3 (pin 11) : UNUSED     (kept attached for wiring parity)
//   servo4 (pin 10) : elbow      (2nd bend)
//   servo5 (pin  9) : wrist / forearm
//   servo6 (pin  8) : wrist tilt
//   servo7 (pin  7) : gripper    (2-finger claw)
//
// Serial protocol (115200 baud, newline-terminated ASCII):
//   Command from laptop:
//     "A a1 a2 a3 a4 a5 a6 a7\n"   set target angles (deg 0..180) for servos 1..7
//                                   use -1 to leave a joint's target unchanged
//     "P\n"                        ping -> replies "PONG"
//     "S\n"                        status -> replies current angles "C a1..a7"
//   Reply to laptop:
//     "OK\n"      command accepted
//     "ERR ...\n" parse error
//     when all joints have reached their targets: "DONE\n" (sent once)
//
// Smoothing: each joint slews toward its target at SLEW_DEG_PER_TICK so the
// arm never snaps. The laptop can send a new target at any time.

#include <Servo.h>

const uint8_t N = 7;
const uint8_t PINS[N] = {13, 12, 11, 10, 9, 8, 7};

// Per-joint safe travel limits (deg). Tighten these once you know the real
// mechanical range so a bad command can't drive a joint into the frame.
const int MIN_DEG[N] = {0, 0, 0, 0, 0, 0, 0};
const int MAX_DEG[N] = {180, 180, 180, 180, 180, 180, 180};

// Startup pose (safe neutral). Matches the old demo's rough home.
const int HOME_DEG[N] = {90, 90, 90, 90, 90, 170, 180};

const uint8_t UNUSED_INDEX = 2;      // servo3, index 2 — attached but never targeted
const float SLEW_DEG_PER_TICK = 1.5; // max move per control tick (~ speed limit)
const int   TICK_MS = 15;            // control loop period
const int   REACH_EPS = 1;           // within this many deg counts as "reached"

Servo servos[N];
float current[N];   // live angle (float for smooth slew)
int   target[N];    // commanded angle
bool  announced;    // have we already sent DONE for the current target set?

int clampJoint(uint8_t i, int deg) {
  if (deg < MIN_DEG[i]) return MIN_DEG[i];
  if (deg > MAX_DEG[i]) return MAX_DEG[i];
  return deg;
}

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < N; i++) {
    servos[i].attach(PINS[i]);
    current[i] = HOME_DEG[i];
    target[i]  = HOME_DEG[i];
    servos[i].write((int)current[i]);
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

// Parse "A a1 a2 ... a7" atomically. -1 keeps the existing target. A malformed
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
    if (i == UNUSED_INDEX) continue;
    float diff = target[i] - current[i];
    if (fabs(diff) <= REACH_EPS) {
      current[i] = target[i];
    } else {
      current[i] += (diff > 0 ? 1 : -1) * min(SLEW_DEG_PER_TICK, (float)fabs(diff));
      allReached = false;
    }
    servos[i].write((int)(current[i] + 0.5));
  }
  if (allReached && !announced) { Serial.println("DONE"); announced = true; }
}

void loop() {
  readSerial();
  slew();
  delay(TICK_MS);
}
