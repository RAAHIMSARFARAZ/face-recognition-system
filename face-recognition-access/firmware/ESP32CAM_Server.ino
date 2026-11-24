#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// =========================
// Wi-Fi
// =========================
const char* ssid = "Your SSID";  //Set these up before flashing your code.
const char* password = "Your Password";

// =========================
// ESP32-CAM IP
// =========================
WebServer server(80);

// =========================
// Onboard Flash LED
// GPIO4 on classic ESP32-CAM
// =========================
#define LED_PIN 4

// =========================
// Camera pins
// =========================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// =========================
// Main webpage
// =========================
void handleRoot() {

  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32-CAM</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>

<body>
  <h1>ESP32-CAM</h1>

  <img id="camera" src="/capture" width="320">

  <br><br>

  <button onclick="ledOn()">LED ON</button>
  <button onclick="ledOff()">LED OFF</button>

  <script>

    function refreshImage() {
      document.getElementById("camera").src =
        "/capture?t=" + new Date().getTime();
    }

    setInterval(refreshImage, 250);

    function ledOn() {
      fetch("/led/on");
    }

    function ledOff() {
      fetch("/led/off");
    }

  </script>
</body>
</html>
)rawliteral";

  server.send(200, "text/html", html);
}

// =========================
// Capture single JPEG
// =========================
void handleCapture() {

  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
    Serial.println("Camera capture failed!");

    server.send(
      500,
      "text/plain",
      "Camera capture failed"
    );

    return;
  }

  server.sendHeader(
    "Cache-Control",
    "no-cache, no-store, must-revalidate"
  );

  server.sendHeader(
    "Pragma",
    "no-cache"
  );

  server.sendHeader(
    "Expires",
    "0"
  );

  server.setContentLength(fb->len);

  server.send(
    200,
    "image/jpeg",
    ""
  );

  WiFiClient client = server.client();

  client.write(
    fb->buf,
    fb->len
  );

  esp_camera_fb_return(fb);
}

// =========================
// LED ON
// =========================
void handleLedOn() {

  digitalWrite(LED_PIN, HIGH);

  Serial.println("LED ON");

  server.send(
    200,
    "text/plain",
    "LED ON"
  );
}

// =========================
// LED OFF
// =========================
void handleLedOff() {

  digitalWrite(LED_PIN, LOW);

  Serial.println("LED OFF");

  server.send(
    200,
    "text/plain",
    "LED OFF"
  );
}

// =========================
// Setup
// =========================
void setup() {

  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("Starting ESP32-CAM...");

  // -------------------------
  // LED setup
  // -------------------------
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // -------------------------
  // Camera configuration
  // -------------------------
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

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  config.pixel_format = PIXFORMAT_JPEG;

  // Stable settings for ESP32-CAM WITHOUT PSRAM
  config.frame_size = FRAMESIZE_HQVGA;
  config.jpeg_quality = 15;
  config.fb_count = 1;

  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_DRAM;

  // -------------------------
  // Initialize camera
  // -------------------------
  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {

    Serial.printf(
      "Camera init failed with error 0x%x\n",
      err
    );

    return;
  }

  Serial.println("Camera initialized successfully!");

  // -------------------------
  // Connect to Wi-Fi
  // -------------------------
  Serial.print("Connecting to Wi-Fi");

  WiFi.begin(
    ssid,
    password
  );

  while (WiFi.status() != WL_CONNECTED) {

    delay(500);

    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected!");

  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  // -------------------------
  // Web server routes
  // -------------------------
  server.on(
    "/",
    HTTP_GET,
    handleRoot
  );

  server.on(
    "/capture",
    HTTP_GET,
    handleCapture
  );

  server.on(
    "/led/on",
    HTTP_GET,
    handleLedOn
  );

  server.on(
    "/led/off",
    HTTP_GET,
    handleLedOff
  );

  // -------------------------
  // Start server
  // -------------------------
  server.begin();

  Serial.println("Web server started!");
  Serial.println();
  Serial.println("Available endpoints:");
  Serial.println("/capture");
  Serial.println("/led/on");
  Serial.println("/led/off");
}

// =========================
// Main loop
// =========================
void loop() {

  server.handleClient();
}