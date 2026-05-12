/*
 * irrigation_inference.ino  FINAL – WITH DEAD‑BAND & ROBUST CALIBRATION
 * =====================================================================
 * Smart Irrigation – ESP32‑S3, TinyML, Lite/Full models
 *
 * NEW FEATURES:
 *  – Raw‑ADC hard override protects against sensor miscalibration
 *  – Target soil moisture bracket (low/high) overrides model at extremes
 *  – Model hysteresis still active inside the bracket
 *  – Soil sensor calibration prints raw ADC for easy setup
 *  – Battery calibration ready (adjust VDIV_RATIO after measuring)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <esp_system.h>
#include <esp_task_wdt.h>

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>

#include <DHT.h>

#include "lite_model.h"
#include "full_model.h"

// ── WiFi ──────────────────────────────────────────
const char* WIFI_SSID     = "Airtel_ashi_0978";   // ← CHANGE
const char* WIFI_PASSWORD = "air34710";            // ← CHANGE
WebServer server(80);

// ── Hardware pins ──────────────────────────────────
#define RELAY_PIN       6
#define STATUS_LED_PIN  38
#define BAT_ADC_PIN     7
#define SOIL_ADC_PIN    5
#define DHT_PIN         4

// ── RELAY POLARITY ─────────────────────────────────
// MOST 5V relay modules are ACTIVE LOW → use LOW.
// If your pump stays OFF when it should be ON, change to HIGH.
#define RELAY_ACTIVE    LOW

// ── Normalisation ──────────────────────────────────
const float NORM_TEMP = 45.0f;
const float NORM_HUM  = 100.0f;
const float NORM_SOIL = 100.0f;

// ── Decision thresholds ────────────────────────────
const float PUMP_ON_THRESHOLD  = 0.55f;     // smoothed probability to turn ON
const float PUMP_OFF_THRESHOLD = 0.35f;     // smoothed probability to turn OFF

// ── Target soil moisture bracket (dead‑band) ───────
// The system will keep moisture in [LOW, HIGH] range.
// Outside this range the model is overridden.
const float TARGET_SOIL_LOW  = 30.0f;       // below this → force pump ON
const float TARGET_SOIL_HIGH = 60.0f;       // above this → force pump OFF

// ── Hard override (raw ADC) ────────────────────────
// If the ADC median is ABOVE this value, the soil is
// considered critically dry (regardless of mapping).
// This is a safety net when calibration is imperfect.
const int RAW_OVERRIDE_DRY = 3800;          // adjust after calibration

// ── Battery & model switching ─────────────────────
// IMPORTANT: after measuring the actual battery voltage
// with a multimeter, adjust VDIV_RATIO so that the
// displayed value matches the true voltage.
const float VDIV_RATIO = 2.0f;              // change if resistors not equal
const float BAT_FULL   = 3.5f;              // voltage to activate FULL model
const float BAT_HYST   = 0.05f;             // hysteresis for switching

// ── Timing ─────────────────────────────────────────
const uint32_t INFERENCE_INTERVAL_MS = 5000;
const uint32_t SENSOR_WARMUP_MS      = 3000;
const uint32_t MAX_IRRIGATION_MS     = 30UL * 60UL * 1000UL;   // 30 min
const uint32_t MIN_RELAY_OFF_MS      = 3UL  * 60UL * 1000UL;   // 3 min ---> turn 1 for testing
const uint8_t  SOIL_FAULT_STREAK_MAX = 6;

// ── Soil calibration (measure & replace) ───────────
// Values from your log: dry = 4095, wet = 1129.
// After you calibrate, set these correctly.
const int SOIL_DRY_RAW = 4000; // ADC value when probe is in air //// changed for my soil sensor from 4095 to 3865 to 3945 to 3995 to 3915 to 4000
const int SOIL_WET_RAW = 1365; // ADC value when probe submerged //// changed for my soil sensor from 1129 to 1435 to 1365 to 1325 to 1385 to 1365

// ── DHT object ─────────────────────────────────────
DHT dht(DHT_PIN, DHT11);

// ── Model descriptors ──────────────────────────────
enum class ModelTier : uint8_t { LITE = 0, FULL = 1 };

struct ModelDescriptor {
  const char*    name;
  const uint8_t* data;
  uint32_t       data_len;
  float          bat_min;
};

const ModelDescriptor MODEL_TABLE[] = {
  { "LITE", g_lite_model_data, g_lite_model_data_len, 0.0f },
  { "FULL", g_full_model_data, g_full_model_data_len, BAT_FULL },
};

// ── Tensor arena & interpreter ─────────────────────
constexpr size_t TENSOR_ARENA_SIZE = 20 * 1024;
alignas(16) uint8_t g_arena[TENSOR_ARENA_SIZE];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::MicroMutableOpResolver<6> resolver;
tflite::MicroInterpreter* interpreter = nullptr;

// ── Runtime state ─────────────────────────────────
ModelTier activeTier = ModelTier::FULL;
bool      modelLoaded = false;

float batFiltered    = 3.8f;
float tempVal        = 0.0f;
float humVal         = 0.0f;
float soilVal        = 0.0f;
float battRaw        = 0.0f;
float rawPrediction  = 0.0f;
float smoothPred     = 0.0f;
bool  pumpState      = false;
bool  hardOverride   = false;
String modelName     = "FULL";
String failReason    = "";

uint32_t lastCycleMs  = 0;
uint32_t cycleCount   = 0;
uint32_t pumpStartMs  = 0;
uint32_t pumpStopMs   = 0;
bool     lastRelaySt  = false;
uint8_t  soilFaultCnt = 0;

// ── Battery reading ───────────────────────────────
float readBatteryVoltage() {
  uint32_t sum = 0;
  for (int i = 0; i < 16; i++) {
    sum += analogRead(BAT_ADC_PIN);
    delayMicroseconds(200);
  }
  float avg = (float)sum / 16.0f;
  float pinV = (avg / 4095.0f) * 3.3f;
  float batV = pinV * VDIV_RATIO;
  return batV;
}

// ── Model selection ───────────────────────────────
ModelTier voltageToTier(float v) {
  return (v >= BAT_FULL) ? ModelTier::FULL : ModelTier::LITE;
}

ModelTier selectModelTier(float batV, ModelTier current) {
  ModelTier desired = voltageToTier(batV);
  if (desired == current) return current;
  bool up = (desired > current);
  float hystV = up ? batV - BAT_HYST : batV + BAT_HYST;
  if (voltageToTier(hystV) == current) return current;
  int next = (int)current + (up ? 1 : -1);
  next = constrain(next, 0, 1);
  return (ModelTier)next;
}

// ── Load model ────────────────────────────────────
bool loadModel(ModelTier tier) {
  esp_task_wdt_reset();
  const auto& desc = MODEL_TABLE[(int)tier];
  const tflite::Model* model = tflite::GetModel(desc.data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERR] %s schema mismatch\n", desc.name);
    return false;
  }
  if (interpreter) { delete interpreter; interpreter = nullptr; }
  interpreter = new tflite::MicroInterpreter(
      model, resolver, g_arena, TENSOR_ARENA_SIZE, error_reporter);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("[ERR] AllocateTensors failed for %s\n", desc.name);
    delete interpreter; interpreter = nullptr;
    return false;
  }
  Serial.printf("[MODEL] %s loaded (arena %u/%u B)\n",
                desc.name, interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);
  modelName = desc.name;
  return true;
}

// ── Sensors ───────────────────────────────────────
bool readSensors() {
  failReason = "";
  float t = NAN, h = NAN;
  for (int attempt = 0; attempt < 3; attempt++) {
    t = dht.readTemperature();
    h = dht.readHumidity();
    if (!isnan(t) && !isnan(h)) break;
    delay(100);
  }
  if (isnan(t) || isnan(h)) {
    failReason = "DHT NaN";
    return false;
  }
  if (t < -40.0f || t > 80.0f || h < 0.0f || h > 100.0f) {
    failReason = "DHT extreme";
    return false;
  }
  tempVal = t;
  humVal  = h;

  // Soil moisture – median filter
  constexpr int N = 15;
  int samples[N];
  for (int i = 0; i < N; i++) {
    samples[i] = analogRead(SOIL_ADC_PIN);
    delay(3);
  }
  for (int i = 0; i < N-1; i++)
    for (int j = i+1; j < N; j++)
      if (samples[j] < samples[i]) {
        int tmp = samples[i]; samples[i] = samples[j]; samples[j] = tmp;
      }
  int median = samples[N/2];

  // Print raw ADC for first 30 cycles (to calibrate)
  if (cycleCount <= 30) {
    Serial.printf("[SOIL] raw median ADC = %d\n", median);
  }

  // Fault detection
  if (median < 100 || median > 4000) {
    soilFaultCnt++;
    if (soilFaultCnt >= SOIL_FAULT_STREAK_MAX) {
      failReason = "Soil ADC stuck";
      Serial.printf("[SENSOR] Soil ADC median %d out of range (fault)\n", median);
      soilVal = 0.0f;   // assume worst case
      return true;      // continue inference
    }
  } else {
    soilFaultCnt = 0;
  }

  // Map to % using calibrated values
  float soil = map(median, SOIL_DRY_RAW, SOIL_WET_RAW, 0, 100);
  soil = constrain(soil, 0.0f, 100.0f);
  soilVal = soil;

  // Raw‑ADC hard override (bypasses all mapping)
  hardOverride = (median > RAW_OVERRIDE_DRY);

  return true;
}

// ── Pump control ──────────────────────────────────
void setPump(bool on) {
  uint32_t now = millis();
  // Watchdog
  if (pumpState && (now - pumpStartMs) >= MAX_IRRIGATION_MS) {
    on = false;
    Serial.println("[WATCHDOG] Max runtime – forced OFF");
  }
  // Minimum off time
  if (!pumpState && on) {
    if ((now - pumpStopMs) < MIN_RELAY_OFF_MS) return;
  }
  if (on == lastRelaySt) return;
  lastRelaySt = on;
  pumpState = on;
  digitalWrite(RELAY_PIN, on ? RELAY_ACTIVE : !RELAY_ACTIVE);
  digitalWrite(STATUS_LED_PIN, on ? HIGH : LOW);
  if (on) {
    pumpStartMs = now;
    Serial.println("[RELAY] ON");
  } else {
    pumpStopMs = now;
    Serial.println("[RELAY] OFF");
  }
}

// ── Inference ─────────────────────────────────────
float runInference() {
  if (!interpreter || !modelLoaded) return -1.0f;

  // Clamp sensor values to training range for normalisation
  float tempClamp = constrain(tempVal, 10.0f, 45.0f);
  float humClamp  = constrain(humVal,  20.0f, 100.0f);
  float soilClamp = constrain(soilVal,  0.0f, 100.0f);

  float features[3] = {
    tempClamp / NORM_TEMP,
    humClamp  / NORM_HUM,
    soilClamp / NORM_SOIL
  };
  for (int i = 0; i < 3; i++) features[i] = constrain(features[i], 0.0f, 1.0f);

  TfLiteTensor* input = interpreter->input(0);
  input->data.f[0] = features[0];
  input->data.f[1] = features[1];
  input->data.f[2] = features[2];

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERR] Invoke failed");
    return -1.0f;
  }
  float prob = interpreter->output(0)->data.f[0];
  return constrain(prob, 0.0f, 1.0f);
}

// ── WiFi dashboard ────────────────────────────────
String buildPage() {
  String html = "<!DOCTYPE html><html><head>";
  html += "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">";
  html += "<meta http-equiv=\"refresh\" content=\"3\">";
  html += "<title>Irrigation AI</title>";
  html += "<style>body{font-family:Arial;text-align:center;background:#f0f0f0;padding:20px;}"
          ".card{background:#fff;border-radius:10px;padding:15px;margin:10px auto;max-width:400px;box-shadow:0 2px 5px rgba(0,0,0,0.1);}"
          ".val{font-size:2em;font-weight:bold;}.on{color:green}.off{color:red}</style>";
  html += "</head><body><h2>Smart Irrigation</h2>";
  html += "<div class='card'>Temperature <div class='val'>" + String(tempVal,1) + "°C</div></div>";
  html += "<div class='card'>Humidity <div class='val'>" + String(humVal,1) + "%</div></div>";
  html += "<div class='card'>Soil Moisture <div class='val'>" + String(soilVal,1) + "%</div></div>";
  html += "<div class='card'>Battery <div class='val'>" + String(battRaw,2) + " V</div></div>";
  html += "<div class='card'>Model <div class='val'>" + modelName + "</div></div>";
  html += "<div class='card'>Prediction <div class='val'>" + String(smoothPred,3) + "</div></div>";
  html += "<div class='card'>Pump <div class='val " + String(pumpState?"on":"off") + "'>" + String(pumpState?"ON":"OFF") + "</div></div>";
  if (failReason != "") {
    html += "<div class='card' style='background:#ffe6e6;'>Sensor Issue: " + failReason + "</div>";
  }
  html += "</body></html>";
  return html;
}
void handleRoot() { server.send(200, "text/html", buildPage()); }

// ── Setup ─────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== AI Irrigation – ESP32-S3 ===");

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, !RELAY_ACTIVE);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  analogReadResolution(12);
  analogSetPinAttenuation(BAT_ADC_PIN, ADC_11db);
  analogSetPinAttenuation(SOIL_ADC_PIN, ADC_11db);

  dht.begin();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println(" connected");
  Serial.println(WiFi.localIP());
  server.on("/", handleRoot);
  server.begin();

  tflite::InitializeTarget();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  battRaw = readBatteryVoltage();
  batFiltered = battRaw;
  Serial.printf("[INIT] Battery: %.3f V\n", battRaw);

  activeTier = voltageToTier(battRaw);
  modelLoaded = loadModel(activeTier);
  if (!modelLoaded) {
    Serial.println("[FATAL] Model load failed – safe blink");
    while (1) {
      digitalWrite(STATUS_LED_PIN, HIGH); delay(100);
      digitalWrite(STATUS_LED_PIN, LOW); delay(100);
    }
  }
  Serial.printf("[INIT] Active: %s\n", MODEL_TABLE[(int)activeTier].name);

  Serial.println("[INIT] Sensor warm-up...");
  delay(SENSOR_WARMUP_MS);
  Serial.println("[INIT] Ready.\n");
}

// ── Main Loop ─────────────────────────────────────
void loop() {
  server.handleClient();
  uint32_t now = millis();
  if (now - lastCycleMs < INFERENCE_INTERVAL_MS) { delay(10); return; }
  lastCycleMs = now;
  cycleCount++;

  battRaw = readBatteryVoltage();
  batFiltered = 0.9f * batFiltered + 0.1f * battRaw;

  ModelTier newTier = selectModelTier(batFiltered, activeTier);
  if (newTier != activeTier) {
    Serial.printf("[SWITCH] %.3fV -> %s\n", batFiltered, MODEL_TABLE[(int)newTier].name);
    if (loadModel(newTier)) {
      activeTier = newTier;
      modelLoaded = true;
    }
  }

  bool sensorsOk = readSensors();

  if (sensorsOk) {
    rawPrediction = runInference();
    if (rawPrediction >= 0.0f) {
      // EMA smoothing
      const float alpha = 0.2f;
      smoothPred = alpha * smoothPred + (1.0f - alpha) * rawPrediction;

      // ── Dead‑band controller ────────────────────
      if (hardOverride) {
        // Raw ADC indicates critically dry → force ON
        setPump(true);
      } else if (soilVal < TARGET_SOIL_LOW) {
        // Below bracket → force ON (safety)
        setPump(true);
      } else if (soilVal > TARGET_SOIL_HIGH) {
        // Above bracket → force OFF (prevent over‑watering)
        setPump(false);
      } else {
        // Inside bracket → use AI hysteresis
        if (smoothPred > PUMP_ON_THRESHOLD) {
          setPump(true);
        } else if (smoothPred < PUMP_OFF_THRESHOLD) {
          setPump(false);
        }
        // else hold current state
      }
    } else {
      setPump(false);
      failReason = "Inference failed";
    }
  } else {
    setPump(false);
    rawPrediction = -1.0f;
    smoothPred = rawPrediction;
  }

  // Telemetry
  Serial.println("--------------------------------");
  Serial.printf("T:%.1fC H:%.1f%% Soil:%.1f%% (raw ovrd:%s) Bat:%.2fV Model:%s Pred:%.3f (smooth:%.3f) Pump:%s",
                tempVal, humVal, soilVal,
                hardOverride ? "YES" : "no",
                battRaw, modelName.c_str(),
                rawPrediction, smoothPred,
                pumpState ? "ON" : "OFF");
  if (!sensorsOk) Serial.printf(" [%s]", failReason.c_str());
  Serial.println();
}
