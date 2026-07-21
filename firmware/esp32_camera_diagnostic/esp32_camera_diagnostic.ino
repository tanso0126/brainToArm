// USB-serial OV2640 diagnostic for the loose ESP32 + camera wiring.
//
// The pin map is the AI-Thinker map supplied with the physical wiring. This
// sketch intentionally uses no Wi-Fi credentials: the Mac requests one frame
// over the CP2102 serial link to prove sensor, wiring, clock, and frame capture.

#include "esp_camera.h"
#include <Wire.h>

constexpr int PWDN_GPIO_NUM = 32;
constexpr int RESET_GPIO_NUM = -1;
constexpr int XCLK_GPIO_NUM = 0;
constexpr int SIOD_GPIO_NUM = 26;
constexpr int SIOC_GPIO_NUM = 27;
constexpr int Y9_GPIO_NUM = 35;
constexpr int Y8_GPIO_NUM = 34;
constexpr int Y7_GPIO_NUM = 39;
constexpr int Y6_GPIO_NUM = 36;
constexpr int Y5_GPIO_NUM = 21;
constexpr int Y4_GPIO_NUM = 19;
constexpr int Y3_GPIO_NUM = 18;
constexpr int Y2_GPIO_NUM = 5;
constexpr int VSYNC_GPIO_NUM = 25;
constexpr int HREF_GPIO_NUM = 23;
constexpr int PCLK_GPIO_NUM = 22;
constexpr int CAMERA_HMIRROR = 0;
constexpr int CAMERA_VFLIP = 0;
constexpr int CAMERA_CAPTURE_XCLK_MHZ = 5;
constexpr int CAMERA_WARMUP_FRAMES = 20;
constexpr framesize_t CAMERA_FRAME_SIZE = FRAMESIZE_QQVGA;

bool cameraReady = false;
esp_err_t cameraError = ESP_OK;
uint16_t cameraPid = 0;
int cameraControlResult = 0;

int discardFrames(int count) {
  int discarded = 0;
  for (int i = 0; i < count; ++i) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame != nullptr) {
      ++discarded;
      esp_camera_fb_return(frame);
    }
    delay(100);
  }
  return discarded;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  Serial.setDebugOutput(false);
  delay(500);
  Serial.println("CAMERA_BOOT");

  // Give a loose OV2640 module a deterministic power cycle. Long jumper wires
  // are also more reliable with SCCB below the usual 100 kHz I2C clock.
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, HIGH);
  delay(100);
  digitalWrite(PWDN_GPIO_NUM, LOW);
  delay(250);
  Wire.begin(SIOD_GPIO_NUM, SIOC_GPIO_NUM, 50000);

  // Zero-initialize the full structure. ESP32 Arduino 2.x added frame-buffer
  // policy fields that must not contain arbitrary stack data.
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  // The connected loose-module harness has camera D5/D7 crossed. Compensate
  // in the GPIO matrix so DMA reconstructs the original byte (and JPEG marker)
  // bit order without requiring the user to disturb the working jumpers.
  config.pin_d5 = Y9_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y7_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = -1;
  config.pin_sccb_scl = -1;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  // QQVGA reduces the parallel transfer volume on the long jumper harness.
  config.frame_size = CAMERA_FRAME_SIZE;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.sccb_i2c_port = 0;

  cameraError = esp_camera_init(&config);
  if (cameraError != ESP_OK) {
    Serial.printf("CAMERA_ERROR 0x%08x\n", cameraError);
    return;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  cameraPid = sensor->id.PID;
  const int frameSizeResult = sensor->set_framesize(sensor, CAMERA_FRAME_SIZE);
  const int xclkResult = sensor->set_xclk(
    sensor, LEDC_TIMER_0, CAMERA_CAPTURE_XCLK_MHZ);
  const int hmirrorResult = sensor->set_hmirror(sensor, CAMERA_HMIRROR);
  const int vflipResult = sensor->set_vflip(sensor, CAMERA_VFLIP);
  Serial.printf("FRAME_SIZE_RESULT %d\n", frameSizeResult);
  Serial.printf("XCLK_RESULT %d\n", xclkResult);
  Serial.printf("CONTROL_RESULT %d\n", cameraControlResult);

  cameraReady = frameSizeResult == 0 && xclkResult == 0 &&
                hmirrorResult == 0 && vflipResult == 0;
  int warmupFrames = 0;
  if (cameraReady) {
    warmupFrames = discardFrames(CAMERA_WARMUP_FRAMES);
  }
  Serial.printf("CAMERA_READY pid=0x%02x psram=%d xclk=%dMHz warmup=%d/%d controls=%d\n",
                cameraPid, psramFound() ? 1 : 0, CAMERA_CAPTURE_XCLK_MHZ,
                warmupFrames, CAMERA_WARMUP_FRAMES, cameraControlResult);
}

uint32_t countTransitions(int pin, uint32_t durationUs) {
  uint32_t transitions = 0;
  int previous = digitalRead(pin);
  const uint32_t start = micros();
  while ((uint32_t)(micros() - start) < durationUs) {
    const int current = digitalRead(pin);
    if (current != previous) {
      ++transitions;
      previous = current;
    }
  }
  return transitions;
}

uint8_t dataTransitionMask(uint32_t durationUs) {
  const int pins[8] = {
    Y2_GPIO_NUM, Y3_GPIO_NUM, Y4_GPIO_NUM, Y5_GPIO_NUM,
    Y6_GPIO_NUM, Y7_GPIO_NUM, Y8_GPIO_NUM, Y9_GPIO_NUM
  };
  int previous[8];
  for (int i = 0; i < 8; ++i) previous[i] = digitalRead(pins[i]);
  uint8_t changed = 0;
  const uint32_t start = micros();
  while ((uint32_t)(micros() - start) < durationUs) {
    for (int i = 0; i < 8; ++i) {
      const int current = digitalRead(pins[i]);
      if (current != previous[i]) {
        changed |= (1U << i);
        previous[i] = current;
      }
    }
  }
  return changed;
}

void sendSignalDiagnostics() {
  const uint32_t pclk = countTransitions(PCLK_GPIO_NUM, 20000);
  const uint32_t href = countTransitions(HREF_GPIO_NUM, 200000);
  const uint32_t vsync = countTransitions(VSYNC_GPIO_NUM, 500000);
  const uint8_t data = dataTransitionMask(200000);
  Serial.printf("SIGNALS pclk=%u href=%u vsync=%u data=0x%02x\n",
                pclk, href, vsync, data);
}

void sendSensorSettings() {
  sensor_t *sensor = esp_camera_sensor_get();
  const camera_status_t &s = sensor->status;
  Serial.printf(
    "SETTINGS wb=%u ae=%d brightness=%d contrast=%d saturation=%d "
    "awb=%u awb_gain=%u aec=%u aec2=%u agc=%u gain=%u ceiling=%u\n",
    s.wb_mode, s.ae_level, s.brightness, s.contrast, s.saturation,
    s.awb, s.awb_gain, s.aec, s.aec2, s.agc, s.agc_gain, s.gainceiling);
}

void tuneSensor(const String &command) {
  int wb = 0;
  int ae = 0;
  int brightness = 0;
  int contrast = 0;
  int saturation = 0;
  if (sscanf(command.c_str(), "TUNE %d %d %d %d %d",
             &wb, &ae, &brightness, &contrast, &saturation) != 5 ||
      wb < 0 || wb > 4 || ae < -2 || ae > 2 ||
      brightness < -2 || brightness > 2 ||
      contrast < -2 || contrast > 2 ||
      saturation < -2 || saturation > 2) {
    Serial.println("TUNE_ERROR expected: TUNE wb(0..4) ae(-2..2) "
                   "brightness(-2..2) contrast(-2..2) saturation(-2..2)");
    return;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  int result = 0;
  result |= sensor->set_whitebal(sensor, 1);
  result |= sensor->set_awb_gain(sensor, 1);
  result |= sensor->set_wb_mode(sensor, wb);
  result |= sensor->set_exposure_ctrl(sensor, 1);
  result |= sensor->set_ae_level(sensor, ae);
  result |= sensor->set_brightness(sensor, brightness);
  result |= sensor->set_contrast(sensor, contrast);
  result |= sensor->set_saturation(sensor, saturation);
  cameraControlResult = result;
  const int settled = discardFrames(CAMERA_WARMUP_FRAMES);
  Serial.printf("TUNE_%s settled=%d/%d\n",
                result == 0 ? "OK" : "ERROR",
                settled, CAMERA_WARMUP_FRAMES);
  sendSensorSettings();
}

void sendFrame() {
  camera_fb_t *frame = nullptr;
  for (int attempt = 0; attempt < 3 && frame == nullptr; ++attempt) {
    frame = esp_camera_fb_get();
    if (frame == nullptr) delay(120);
  }
  if (frame == nullptr) {
    Serial.println("CAPTURE_ERROR");
    return;
  }

  Serial.printf("FRAME %u %u %u %u\n",
                frame->width, frame->height, frame->len, frame->format);
  Serial.write(frame->buf, frame->len);
  Serial.print("\nFRAME_END\n");
  esp_camera_fb_return(frame);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "STATUS") {
      if (cameraReady) {
        Serial.printf("CAMERA_READY pid=0x%02x psram=%d controls=%d\n",
                      cameraPid, psramFound() ? 1 : 0, cameraControlResult);
      } else {
        Serial.printf("CAMERA_ERROR 0x%08x\n", cameraError);
      }
    } else if (command == "SIGNALS" && cameraReady) {
      sendSignalDiagnostics();
    } else if (command == "SETTINGS" && cameraReady) {
      sendSensorSettings();
    } else if (command.startsWith("TUNE ") && cameraReady) {
      tuneSensor(command);
    } else if (command == "CAPTURE" && cameraReady) {
      sendFrame();
    }
  }
  delay(10);
}
