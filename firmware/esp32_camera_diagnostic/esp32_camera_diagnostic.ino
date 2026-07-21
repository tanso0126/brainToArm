// USB-serial OV2640 diagnostic for the loose ESP32 + camera wiring.
//
// The pin map is the AI-Thinker map supplied with the physical wiring. This
// sketch intentionally uses no Wi-Fi credentials: the Mac requests one JPEG
// over the CP2102 serial link to prove sensor, wiring, clock, and frame capture.

#include "esp_camera.h"

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

bool cameraReady = false;
esp_err_t cameraError = ESP_OK;
uint16_t cameraPid = 0;

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  delay(500);
  Serial.println("CAMERA_BOOT");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  // This core/sensor combination probes reliably at VGA. Once initialized,
  // lower it to QVGA so the no-PSRAM DevKit has ample internal frame memory.
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = psramFound() ? 2 : 1;

  cameraError = esp_camera_init(&config);
  if (cameraError != ESP_OK) {
    Serial.printf("CAMERA_ERROR 0x%08x\n", cameraError);
    return;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  cameraPid = sensor->id.PID;
  const int frameSizeResult = sensor->set_framesize(sensor, FRAMESIZE_QVGA);
  Serial.printf("FRAME_SIZE_RESULT %d\n", frameSizeResult);

  // Let exposure/white balance settle and discard stale startup frames.
  int warmupFrames = 0;
  for (int i = 0; i < 4; ++i) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame != nullptr) {
      ++warmupFrames;
      esp_camera_fb_return(frame);
    }
    delay(120);
  }

  cameraReady = frameSizeResult == 0;
  Serial.printf("CAMERA_READY pid=0x%02x psram=%d warmup=%d/4 xclk=20MHz\n",
                cameraPid, psramFound() ? 1 : 0, warmupFrames);
}

void sendFrame() {
  camera_fb_t *frame = nullptr;
  for (int attempt = 0; attempt < 8 && frame == nullptr; ++attempt) {
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
    const int command = Serial.read();
    if (command == '?') {
      if (cameraReady) {
        Serial.printf("CAMERA_READY pid=0x%02x psram=%d\n",
                      cameraPid, psramFound() ? 1 : 0);
      } else {
        Serial.printf("CAMERA_ERROR 0x%08x\n", cameraError);
      }
    } else if ((command == 'C' || command == 'c') && cameraReady) {
      sendFrame();
    }
    while (Serial.available()) Serial.read();
  }
  delay(10);
}
