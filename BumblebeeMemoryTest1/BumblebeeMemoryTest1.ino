#include <Arduino.h>
#include <WiFi.h>
#include <ESP_I2S.h>
#include <Chirale_TensorFlowLite.h>
#include <LiquidCrystal.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

#include "model.h"
#include "config.h"

// ============================================================
// Wi-Fi & WebSocket Server Configuration
// ============================================================

const char* WIFI_SSID     = BUMBLEBEE_WIFI_SSID;
const char* WIFI_PASSWORD = BUMBLEBEE_WIFI_PASSWORD;
const char* WS_HOST       = BUMBLEBEE_WS_HOST;
constexpr uint16_t WS_PORT = 8765;

// Memory benchmarks: track Wi-Fi vs AI Model footprints
size_t g_heap_boot        = 0;
size_t g_heap_before_wifi = 0;
size_t g_wifi_ram_used    = 0;

// ============================================================
// Pins
// ============================================================

#define BCLK_PIN 18
#define WS_PIN   19
#define DIN_PIN  23

// LCD: RS=21, E=22, D4=4, D5=5, D6=33, D7=2
LiquidCrystal lcd(21, 22, 4, 5, 33, 2);

I2SClass I2S;

// ============================================================
// Audio Buffers
// RAM reduction: raw_buffer 256->128 samples (saves 512 B static DRAM)
//                pcm_buffer 256->128 samples (saves 256 B static DRAM)
// ============================================================

constexpr int SAMPLE_RATE        = 16000;
constexpr int I2S_BUFFER_SAMPLES = 256;
constexpr int FRAME_SAMPLES      = 160;

int32_t raw_buffer[I2S_BUFFER_SAMPLES];
int16_t pcm_buffer[I2S_BUFFER_SAMPLES];

// Pre-roll ring buffer: 250 ms  (4000 x 2 B = 8 KB heap)
// Was 500 ms / 8000 samples / 16 KB — saves 8 KB heap.
constexpr size_t PREROLL_SAMPLES = 4000;
int16_t* preroll_buffer          = nullptr;
size_t   preroll_write_idx       = 0;
bool     preroll_wrapped         = false;

void pushPreroll(const int16_t* samples, size_t count) {
  if (!preroll_buffer) return;
  for (size_t i = 0; i < count; ++i) {
    preroll_buffer[preroll_write_idx++] = samples[i];
    if (preroll_write_idx >= PREROLL_SAMPLES) {
      preroll_write_idx = 0;
      preroll_wrapped   = true;
    }
  }
}

// Reset pre-roll so ASR audio doesn't bleed into the next session
void clearPreroll() {
  if (!preroll_buffer) return;
  memset(preroll_buffer, 0, PREROLL_SAMPLES * sizeof(int16_t));
  preroll_write_idx = 0;
  preroll_wrapped   = false;
}

// ============================================================
// Lightweight Self-Contained WebSocket Client (RFC 6455)
// Uses built-in WiFiClient — zero external library dependencies!
// ============================================================

class TinyWebSocketClient {
private:
  WiFiClient client;
  bool connected = false;

  void sendMaskedFrame(uint8_t opcode, const uint8_t* payload, size_t length) {
    if (!client.connected()) {
      connected = false;
      return;
    }

    uint8_t mask[4] = {
      (uint8_t)random(1, 255),
      (uint8_t)random(1, 255),
      (uint8_t)random(1, 255),
      (uint8_t)random(1, 255)
    };

    uint8_t b0 = 0x80 | (opcode & 0x0F);
    client.write(b0);

    if (length <= 125) {
      client.write((uint8_t)(0x80 | (uint8_t)length));
    } else if (length <= 65535) {
      client.write((uint8_t)(0x80 | 126));
      client.write((uint8_t)((length >> 8) & 0xFF));
      client.write((uint8_t)(length & 0xFF));
    } else {
      return;
    }

    client.write(mask, 4);

    // Stream masked payload in 64-byte bursts to keep stack usage low
    uint8_t chunk[64];
    size_t sent = 0;
    while (sent < length) {
      size_t batch = length - sent;
      if (batch > sizeof(chunk)) batch = sizeof(chunk);
      for (size_t i = 0; i < batch; ++i) {
        chunk[i] = payload[sent + i] ^ mask[(sent + i) % 4];
      }
      client.write(chunk, batch);
      sent += batch;
    }
  }

public:
  bool connect(const char* host, uint16_t port) {
    // Always make a fresh TCP connection — never reuse a potentially dead socket
    if (client.connected()) {
      client.stop();
    }
    connected = false;

    Serial.print("[WS] Connecting to ws://");
    Serial.print(host);
    Serial.print(":");
    Serial.println(port);

    if (!client.connect(host, port)) {
      Serial.println("[WS] TCP connection failed.");
      return false;
    }

    // HTTP Upgrade handshake
    client.print("GET / HTTP/1.1\r\n");
    client.print("Host: "); client.print(host); client.print(":"); client.print(port); client.print("\r\n");
    client.print("Upgrade: websocket\r\n");
    client.print("Connection: Upgrade\r\n");
    client.print("Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n");
    client.print("Sec-WebSocket-Version: 13\r\n");
    client.print("\r\n");

    // Read handshake response
    unsigned long start = millis();
    String status_line  = "";
    while (client.connected() && millis() - start < 3000) {
      if (client.available()) {
        status_line = client.readStringUntil('\n');
        break;
      }
      delay(5);
    }

    if (!status_line.startsWith("HTTP/1.1 101")) {
      Serial.print("[WS] Handshake failed: ");
      Serial.println(status_line);
      client.stop();
      return false;
    }

    // Drain remaining HTTP headers
    while (client.connected() && millis() - start < 3000) {
      if (client.available()) {
        String line = client.readStringUntil('\n');
        if (line == "\r" || line.length() == 0) break;
      }
      delay(2);
    }

    connected = true;
    Serial.println("[WS] Handshake OK (101 Switching Protocols).");
    return true;
  }

  bool isConnected() {
    if (!client.connected()) {
      if (connected) {
        Serial.println("[WS] Socket closed by remote host.");
      }
      connected = false;
    }
    return connected;
  }

  void sendText(const char* text) {
    if (!isConnected()) return;
    sendMaskedFrame(0x01, reinterpret_cast<const uint8_t*>(text), strlen(text));
  }

  void sendBinary(const uint8_t* data, size_t length) {
    if (!isConnected()) return;
    sendMaskedFrame(0x02, data, length);
  }

  // Wait for a single text frame reply from the server
  bool readTextReply(String& out_text, unsigned long timeout_ms = 3000) {
    unsigned long start = millis();
    while (client.connected() && millis() - start < timeout_ms) {
      if (client.available() >= 2) {
        uint8_t b0 = client.read();
        uint8_t b1 = client.read();

        uint8_t opcode = b0 & 0x0F;
        bool    masked = (b1 & 0x80) != 0;
        size_t  len    = (b1 & 0x7F);

        if (len == 126) {
          while (client.available() < 2 && millis() - start < timeout_ms) delay(1);
          len = ((size_t)client.read() << 8) | client.read();
        }

        uint8_t mask[4] = {0, 0, 0, 0};
        if (masked) {
          while (client.available() < 4 && millis() - start < timeout_ms) delay(1);
          client.readBytes(mask, 4);
        }

        out_text = "";
        out_text.reserve(len);
        for (size_t i = 0; i < len; ++i) {
          while (!client.available() && millis() - start < timeout_ms) delay(1);
          uint8_t b = client.read();
          if (masked) b ^= mask[i % 4];
          out_text += (char)b;
        }

        if (opcode == 0x01) return true;  // Text frame
        if (opcode == 0x08) {             // Server sent Close frame
          Serial.println("[WS] Server sent Close frame.");
          close();
          return false;
        }
      }
      delay(5);
    }
    return false;
  }

  // Send a proper WebSocket Close frame and stop the TCP socket
  void close() {
    if (client.connected()) {
      uint8_t close_frame[2] = {0x88, 0x00};
      client.write(close_frame, 2);
      delay(50);   // give server a moment to ack
      client.stop();
    }
    connected = false;
    Serial.println("[WS] WebSocket closed.");
  }
};

TinyWebSocketClient wsClient;

// ============================================================
// TFLite
// ============================================================

// Model actually requires 20,952 bytes (20.46 KB).
// Allocating 24 KB provides ~3.5 KB safe headroom while saving 40 KB of RAM!
constexpr int kTensorArenaSize = 24 * 1024;
uint8_t* tensor_arena = nullptr;

const tflite::Model*            model              = nullptr;
tflite::MicroInterpreter*       interpreter        = nullptr;
tflite::MicroResourceVariables* resource_variables = nullptr;

TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

tflite::MicroMutableOpResolver<14> resolver;

// ============================================================
// Microfrontend
// ============================================================

FrontendState  frontend_state;
FrontendConfig frontend_config;

int8_t feature_buffer[3][40];
int    feature_frames = 0;

// ============================================================
// Detection
// ============================================================

constexpr float         DETECTION_THRESHOLD  = 0.55f;   // Reduced from 0.65/0.70 for fast, easy triggering
constexpr int           REQUIRED_CONSECUTIVE = 2;       // 2 consecutive hits (20 ms) for instant response
constexpr unsigned long COOLDOWN_MS          = 1500;

int           consecutive_hits = 0;
unsigned long cooldown_until   = 0;

// ============================================================
// LCD helpers
// ============================================================

void lcdClearAndRefresh() {
  lcd.clear();
  delay(5);
  lcd.setCursor(0, 0); lcd.print("                ");
  lcd.setCursor(0, 1); lcd.print("                ");
  lcd.setCursor(0, 0);
}

void lcdMessage(const char* line1, const char* line2) {
  lcdClearAndRefresh();
  lcd.setCursor(0, 0); lcd.print(line1);
  lcd.setCursor(0, 1); lcd.print(line2);
}

void lcdMessageDelay(const char* line1, const char* line2, unsigned long delay_ms = 2000) {
  lcdMessage(line1, line2);
  delay(delay_ms);
}

// ============================================================
// Feature quantization
// ============================================================

int8_t quantizeFeature(uint16_t feature) {
  constexpr int32_t value_scale = 256;
  constexpr int32_t value_div   = 666;
  int32_t value = ((int32_t(feature) * value_scale) + (value_div / 2)) / value_div;
  value -= 128;
  if (value < -128) value = -128;
  if (value >  127) value =  127;
  return static_cast<int8_t>(value);
}

// ============================================================
// Microfrontend initialization
// ============================================================

bool initializeFrontend() {
  frontend_config.window.size_ms              = 30;
  frontend_config.window.step_size_ms         = 10;

  frontend_config.filterbank.num_channels     = 40;
  frontend_config.filterbank.lower_band_limit = 125;
  frontend_config.filterbank.upper_band_limit = 7500;

  frontend_config.noise_reduction.smoothing_bits       = 10;
  frontend_config.noise_reduction.even_smoothing       = 0.025;
  frontend_config.noise_reduction.odd_smoothing        = 0.06;
  frontend_config.noise_reduction.min_signal_remaining = 0.05;

  frontend_config.pcan_gain_control.enable_pcan = 1;
  frontend_config.pcan_gain_control.strength    = 0.95;
  frontend_config.pcan_gain_control.offset      = 80.0;
  frontend_config.pcan_gain_control.gain_bits   = 21;

  frontend_config.log_scale.enable_log  = 1;
  frontend_config.log_scale.scale_shift = 6;

  return FrontendPopulateState(&frontend_config, &frontend_state, SAMPLE_RATE);
}

// ============================================================
// TFLite initialization
// ============================================================

bool initializeTFLite() {
  Serial.println("Registering TFLite operators...");

  if (resolver.AddCallOnce()        != kTfLiteOk) return false;
  if (resolver.AddConcatenation()   != kTfLiteOk) return false;
  if (resolver.AddConv2D()          != kTfLiteOk) return false;
  if (resolver.AddDepthwiseConv2D() != kTfLiteOk) return false;
  if (resolver.AddFullyConnected()  != kTfLiteOk) return false;
  if (resolver.AddLogistic()        != kTfLiteOk) return false;
  if (resolver.AddReshape()         != kTfLiteOk) return false;
  if (resolver.AddStridedSlice()    != kTfLiteOk) return false;
  if (resolver.AddMean()            != kTfLiteOk) return false;
  if (resolver.AddQuantize()        != kTfLiteOk) return false;
  if (resolver.AddVarHandle()       != kTfLiteOk) return false;
  if (resolver.AddReadVariable()    != kTfLiteOk) return false;
  if (resolver.AddAssignVariable()  != kTfLiteOk) return false;
  if (resolver.AddSplitV()          != kTfLiteOk) return false;

  Serial.println("Operators registered.");

  model = tflite::GetModel(bumblebee_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Schema version mismatch!");
    return false;
  }

  // tensor_arena is already malloc'd in setup()
  static tflite::MicroAllocator* allocator = nullptr;
  allocator = tflite::MicroAllocator::Create(tensor_arena, kTensorArenaSize);
  if (!allocator) { Serial.println("ERROR: Allocator failed."); return false; }

  resource_variables = tflite::MicroResourceVariables::Create(allocator, 8);
  if (!resource_variables) { Serial.println("ERROR: ResourceVariables failed."); return false; }

  static tflite::MicroInterpreter static_interpreter(model, resolver, allocator, resource_variables);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() FAILED!");
    return false;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);
  if (!input || !output) { Serial.println("ERROR: Tensors missing."); return false; }

  size_t used_arena = interpreter->arena_used_bytes();
  Serial.print("[TFLite] Arena allocated: ");
  Serial.print(kTensorArenaSize);
  Serial.print(" bytes (");
  Serial.print(static_cast<float>(kTensorArenaSize) / 1024.0f, 2);
  Serial.print(" KB) | Actually used: ");
  Serial.print(used_arena);
  Serial.print(" bytes (");
  Serial.print(static_cast<float>(used_arena) / 1024.0f, 2);
  Serial.print(" KB) | Unused headroom: ");
  Serial.print(kTensorArenaSize - used_arena);
  Serial.println(" bytes");

  Serial.println("TFLite OK.");
  return true;
}

// ============================================================
// Run KWS inference
// ============================================================

bool runKWS() {
  if (!interpreter || !input || !output) return false;
  if (input->type != kTfLiteInt8)        return false;
  if (input->bytes < sizeof(feature_buffer)) return false;

  memcpy(input->data.int8, feature_buffer, sizeof(feature_buffer));
  if (interpreter->Invoke() != kTfLiteOk) return false;

  float score = 0.0f;
  if (output->type == kTfLiteUInt8) {
    score = (output->data.uint8[0] - output->params.zero_point) * output->params.scale;
  } else if (output->type == kTfLiteInt8) {
    score = (output->data.int8[0]  - output->params.zero_point) * output->params.scale;
  } else {
    return false;
  }

  if (score >= DETECTION_THRESHOLD) {
    consecutive_hits++;
  } else {
    consecutive_hits = 0;
  }

  if (score >= 0.35f) {
    Serial.print("[KWS] score: ");
    Serial.print(score, 2);
    Serial.print(" | hits: ");
    Serial.print(consecutive_hits);
    Serial.print("/");
    Serial.println(REQUIRED_CONSECUTIVE);
  }

  if (consecutive_hits >= REQUIRED_CONSECUTIVE) {
    consecutive_hits = 0;
    if (millis() >= cooldown_until) {
      return true;
    }
  }
  return false;
}

// ============================================================
// Process one frontend feature frame
// ============================================================

void recordAndSendToASR();   // forward declaration
void printFinalRAMUsage();   // forward declaration

void processFeature(const FrontendOutput& frontend_output) {
  if (frontend_output.size != 40) return;

  if (feature_frames < 3) {
    for (int i = 0; i < 40; i++) {
      feature_buffer[feature_frames][i] = quantizeFeature(frontend_output.values[i]);
    }
    feature_frames++;
  } else {
    memcpy(feature_buffer[0], feature_buffer[1], 40);
    memcpy(feature_buffer[1], feature_buffer[2], 40);
    for (int i = 0; i < 40; i++) {
      feature_buffer[2][i] = quantizeFeature(frontend_output.values[i]);
    }
  }

  if (feature_frames >= 3 && runKWS()) {
    Serial.println();
    Serial.println("==============================");
    Serial.println("BUMBLEBEE DETECTED");
    Serial.println("==============================");
    lcdMessage("Bumblebee", "Detected");
    recordAndSendToASR();
  }
}

// ============================================================
// Process PCM samples through the microfrontend
// ============================================================

void processPCM(int16_t* samples, size_t count) {
  size_t processed = 0;
  while (processed < count) {
    size_t samples_read = 0;
    FrontendOutput out = FrontendProcessSamples(
        &frontend_state, samples + processed, count - processed, &samples_read);
    if (samples_read == 0) break;
    processed += samples_read;
    if (out.size > 0) processFeature(out);
  }
}

// ============================================================
// Reset KWS state
// ============================================================

void resetKWSState() {
  memset(feature_buffer, 0, sizeof(feature_buffer));
  feature_frames   = 0;
  consecutive_hits = 0;
}



// ============================================================
// VAD helpers
// ============================================================

constexpr unsigned long MAX_ASR_DURATION_MS = 12000;
constexpr unsigned long END_SILENCE_MS      = 1800;
constexpr uint32_t      MIN_NOISE_THRESHOLD = 500;
constexpr float         NOISE_MULTIPLIER    = 2.0f;

uint32_t calculateAudioEnergy(const int16_t* samples, size_t count) {
  if (!samples || count == 0) return 0;
  uint64_t sum = 0;
  for (size_t i = 0; i < count; ++i) {
    int32_t v = samples[i];
    if (v < 0) v = -v;
    sum += (uint32_t)v;
  }
  return (uint32_t)(sum / count);
}

String extractJsonText(const String& body) {
  int key_start = body.indexOf("\"text\"");
  if (key_start < 0) return "";
  int colon = body.indexOf(':', key_start + 6);
  if (colon  < 0) return "";
  int q1    = body.indexOf('"', colon + 1);
  if (q1    < 0) return "";
  int q2    = body.indexOf('"', q1 + 1);
  if (q2    < 0) return "";
  return body.substring(q1 + 1, q2);
}

// ============================================================
// Live stream microphone audio to Vosk over WebSocket
//
// FIX 1 — WebSocket is explicitly closed at the END of every
//          session.  Next call always opens a fresh connection
//          so there is no stale-socket connect/disconnect loop.
//
// FIX 2 — Cooldown timer set AFTER ASR completes so KWS
//          re-arms only after we are back in listen mode.
//
// FIX 3 — After returning: flushI2SDMA() discards stale DMA
//          frames; FrontendPopulateState() re-inits noise floor
//          so KWS detection accuracy recovers immediately.
//
// RAM savings vs Bumblebee_WebSocket.ino:
//   asr_stream_buffer (1 KB) is now stack-local inside this
//   function — freed automatically on return, not occupying
//   global DRAM during idle KWS operation.
// ============================================================

void recordAndSendToASR() {

  // Always open a fresh connection per session.
  // The server closes its handler after sending the END reply so
  // the previous socket is already dead on entry.
  if (wsClient.isConnected()) {
    wsClient.close();
  }

  if (!wsClient.connect(WS_HOST, WS_PORT)) {
    Serial.println("[ASR] ERROR: Cannot connect to WebSocket server.");
    lcdMessage("ASR", "WS Error");
    delay(1000);
    resetKWSState();
    lcdMessage("Bumblebee", "Listening");
    return;
  }

  // 1. START frame
  wsClient.sendText("START");
  lcdMessage("Speech", "Listening...");

  // asr_stream_buffer lives on the stack for the duration of this
  // function only (~5-12 s max), saving 1 KB of persistent global DRAM.
  constexpr size_t ASR_STREAM_CHUNK_BYTES = 1024;
  uint8_t asr_stream_buffer[ASR_STREAM_CHUNK_BYTES];

  // 2. Stream pre-roll (250 ms of audio captured before wake-word)
  {
    size_t start_pos         = preroll_wrapped ? preroll_write_idx : 0;
    size_t total_preroll     = preroll_wrapped ? PREROLL_SAMPLES   : preroll_write_idx;
    size_t sent_preroll      = 0;
    const  size_t max_samples = ASR_STREAM_CHUNK_BYTES / sizeof(int16_t);

    while (sent_preroll < total_preroll) {
      size_t chunk_samples = total_preroll - sent_preroll;
      if (chunk_samples > max_samples) chunk_samples = max_samples;

      int16_t temp_buf[max_samples];
      for (size_t i = 0; i < chunk_samples; ++i) {
        size_t read_idx = (start_pos + sent_preroll + i) % PREROLL_SAMPLES;
        temp_buf[i]     = preroll_buffer[read_idx];
      }
      wsClient.sendBinary(reinterpret_cast<const uint8_t*>(temp_buf),
                          chunk_samples * sizeof(int16_t));
      sent_preroll += chunk_samples;
    }
  }

  // 3. Continuous real-time capture until silence or max duration
  unsigned long session_start    = millis();
  unsigned long low_energy_start = 0;
  float         noise_floor      = (float)MIN_NOISE_THRESHOLD;
  size_t        buffered_bytes   = 0;

  while (millis() - session_start < MAX_ASR_DURATION_MS) {

    int bytes_read = I2S.readBytes((char*)raw_buffer, sizeof(raw_buffer));
    if (bytes_read <= 0) continue;

    int samples_available = bytes_read / sizeof(int32_t);
    if (samples_available > I2S_BUFFER_SAMPLES) samples_available = I2S_BUFFER_SAMPLES;
    if (samples_available <= 0) continue;

    for (int i = 0; i < samples_available; ++i) {
      int64_t v = ((int64_t)raw_buffer[i]) >> 14;
      if (v >  32767) v =  32767;
      if (v < -32768) v = -32768;
      pcm_buffer[i] = (int16_t)v;
    }

    size_t samples_to_process = (size_t)samples_available;

    uint32_t      energy = calculateAudioEnergy(pcm_buffer, samples_to_process);
    unsigned long now    = millis();

    uint32_t dynamic_threshold = (uint32_t)(noise_floor * NOISE_MULTIPLIER);
    if (dynamic_threshold < MIN_NOISE_THRESHOLD) dynamic_threshold = MIN_NOISE_THRESHOLD;

    if (energy < dynamic_threshold) {
      if (low_energy_start == 0) low_energy_start = now;
      noise_floor = noise_floor * 0.90f + (float)energy * 0.10f;
      if (now - low_energy_start >= END_SILENCE_MS) break;
    } else {
      low_energy_start = 0;
    }

    // Pack into 1 KB chunks and send
    size_t bytes_to_copy = samples_to_process * sizeof(int16_t);
    size_t offset        = 0;

    while (offset < bytes_to_copy) {
      size_t space      = ASR_STREAM_CHUNK_BYTES - buffered_bytes;
      size_t copy_bytes = bytes_to_copy - offset;
      if (copy_bytes > space) copy_bytes = space;

      memcpy(asr_stream_buffer + buffered_bytes,
             reinterpret_cast<uint8_t*>(pcm_buffer) + offset,
             copy_bytes);
      buffered_bytes += copy_bytes;
      offset         += copy_bytes;

      if (buffered_bytes == ASR_STREAM_CHUNK_BYTES) {
        wsClient.sendBinary(asr_stream_buffer, buffered_bytes);
        buffered_bytes = 0;
      }
    }
  }

  // 4. Flush remaining bytes
  if (buffered_bytes > 0) {
    wsClient.sendBinary(asr_stream_buffer, buffered_bytes);
  }

  // 5. END frame — server will finalise transcription and send reply
  wsClient.sendText("END");

  // 6. Wait for the server's JSON reply
  String reply = "";
  if (wsClient.readTextReply(reply, 3000)) {
    String text = extractJsonText(reply);
    Serial.print("[ASR] Transcription: ");
    Serial.println(text);

    if (text.length() > 0) {
      lcdMessage("ASR Result:", text.substring(0, 16).c_str());
      delay(2000);
    } else {
      lcdMessage("ASR", "No speech");
      delay(1000);
    }
  } else {
    Serial.println("[ASR] No reply or timeout.");
    lcdMessage("ASR", "Complete");
    delay(1000);
  }

  // 7. Close WebSocket cleanly at end of every session.
  //    The server handler has already exited after sending the reply.
  //    Closing here keeps both sides in sync and prevents the
  //    connect/disconnect loop on the next wake-word trigger.
  wsClient.close();

  // 8. Reset KWS feature buffer
  resetKWSState();

  // 9. Clear pre-roll so ASR audio doesn't bleed into next session
  clearPreroll();

  // 10. Cooldown starts NOW: 1.5s is plenty to avoid self-triggering
  //     without locking out the user for 7 full seconds!
  cooldown_until = millis() + COOLDOWN_MS;

  lcdMessage("Bumblebee", "Listening");
  Serial.println("[KWS] Listening resumed.");

  // Print final RAM usage diagnostics
  printFinalRAMUsage();
}

// ============================================================
// RAM Diagnostics
// ============================================================

void printFinalRAMUsage() {
  const size_t audio_buffer_bytes =
      sizeof(raw_buffer) + sizeof(pcm_buffer) + (PREROLL_SAMPLES * sizeof(int16_t));
  const size_t asr_stream_buffer_bytes = 1024;  // chunk size used during streaming
  const size_t feature_buffer_bytes    = sizeof(feature_buffer);
  const size_t tensor_arena_bytes      = kTensorArenaSize;
  const size_t arena_used_bytes        = interpreter ? interpreter->arena_used_bytes() : 0;

  // Pure Application & AI Model footprint
  const size_t app_ai_model_bytes      = audio_buffer_bytes + asr_stream_buffer_bytes +
                                         feature_buffer_bytes + tensor_arena_bytes;

  const size_t heap_total        = ESP.getHeapSize();
  const size_t heap_free         = ESP.getFreeHeap();
  const size_t current_heap_used = (heap_total > heap_free) ? (heap_total - heap_free) : 0;
  const size_t minimum_free_heap = ESP.getMinFreeHeap();
  const size_t peak_heap_used    = (heap_total > minimum_free_heap) ? (heap_total - minimum_free_heap) : 0;

  // Compulsory system & network allocations
  size_t app_heap_allocations = (PREROLL_SAMPLES * sizeof(int16_t)) + kTensorArenaSize;
  size_t system_compulsory_heap = (peak_heap_used > app_heap_allocations) ? (peak_heap_used - app_heap_allocations) : 0;
  size_t other_system_ram = (system_compulsory_heap > g_wifi_ram_used) ? (system_compulsory_heap - g_wifi_ram_used) : 0;

  const size_t peak_measured_ram = app_ai_model_bytes + peak_heap_used;

  Serial.println();
  Serial.println("==========================================================");
  Serial.println("        BUMBLEBEE RAM BENCHMARK BREAKDOWN");
  Serial.println("==========================================================");
  Serial.println();
  Serial.println("[1] ON-DEVICE AI MODEL & AUDIO (OUR APPLICATION FOOTPRINT)");
  Serial.println("----------------------------------------------------------");
  Serial.print("  Audio buffers (I2S + 250ms Pre-roll) : ");
  Serial.print(audio_buffer_bytes);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(audio_buffer_bytes) / 1024.0f, 2);
  Serial.println(" KB)");

  Serial.print("  Microfrontend feature buffer         : ");
  Serial.print(feature_buffer_bytes);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(feature_buffer_bytes) / 1024.0f, 2);
  Serial.println(" KB)");

  Serial.print("  ASR streaming chunk buffer           : ");
  Serial.print(asr_stream_buffer_bytes);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(asr_stream_buffer_bytes) / 1024.0f, 2);
  Serial.println(" KB)");

  Serial.print("  TensorFlow arena (allocated)         : ");
  Serial.print(tensor_arena_bytes);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(tensor_arena_bytes) / 1024.0f, 2);
  Serial.println(" KB)");

  if (arena_used_bytes > 0) {
    Serial.print("    -> Actual TFLite model weights/ops : ");
    Serial.print(arena_used_bytes);
    Serial.print(" B  (");
    Serial.print(static_cast<float>(arena_used_bytes) / 1024.0f, 2);
    Serial.println(" KB)");

    Serial.print("    -> Unused arena safety margin      : ");
    Serial.print(tensor_arena_bytes - arena_used_bytes);
    Serial.print(" B  (");
    Serial.print(static_cast<float>(tensor_arena_bytes - arena_used_bytes) / 1024.0f, 2);
    Serial.println(" KB)");
  }

  Serial.println("----------------------------------------------------------");
  Serial.print("  >>> TOTAL AI MODEL & AUDIO FOOTPRINT : ");
  Serial.print(app_ai_model_bytes);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(app_ai_model_bytes) / 1024.0f, 2);
  Serial.println(" KB) <<<");
  Serial.println();

  Serial.println("[2] COMPULSORY HARDWARE & SYSTEM NETWORKING (FIRMWARE)");
  Serial.println("----------------------------------------------------------");
  Serial.print("  Wi-Fi Driver & lwIP TCP/IP Stack     : ");
  Serial.print(g_wifi_ram_used);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(g_wifi_ram_used) / 1024.0f, 2);
  Serial.println(" KB)  [Silicon compulsory]");

  Serial.print("  FreeRTOS Kernel, Tasks & System Heap : ");
  Serial.print(other_system_ram);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(other_system_ram) / 1024.0f, 2);
  Serial.println(" KB)");

  Serial.println("----------------------------------------------------------");
  Serial.print("  >>> TOTAL SYSTEM & WI-FI OVERHEAD    : ");
  Serial.print(system_compulsory_heap);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(system_compulsory_heap) / 1024.0f, 2);
  Serial.println(" KB) <<<");
  Serial.println();

  Serial.println("[3] OVERALL RUNTIME MEMORY POOL");
  Serial.println("----------------------------------------------------------");
  Serial.print("  ESP32 SRAM pool total                : ");
  Serial.print(heap_total);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(heap_total) / 1024.0f, 2);
  Serial.println(" KB)");

  Serial.print("  Current heap used                    : ");
  Serial.print(current_heap_used);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(current_heap_used) / 1024.0f, 2);
  Serial.println(" KB)");

  Serial.print("  Remaining free heap                  : ");
  Serial.print(heap_free);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(heap_free) / 1024.0f, 2);
  Serial.println(" KB)");

  Serial.print("  Peak heap used                       : ");
  Serial.print(peak_heap_used);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(peak_heap_used) / 1024.0f, 2);
  Serial.println(" KB)");

  Serial.print("  Total Peak Measured RAM (App + OS)   : ");
  Serial.print(peak_measured_ram);
  Serial.print(" B  (");
  Serial.print(static_cast<float>(peak_measured_ram) / 1024.0f, 2);
  Serial.println(" KB)");
  Serial.println("==========================================================");
  Serial.println();
}

// ============================================================
// I2S initialization
// ============================================================

bool initializeI2S() {
  I2S.setPins(BCLK_PIN, WS_PIN, -1, DIN_PIN);
  if (!I2S.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("I2S initialization FAILED.");
    return false;
  }
  Serial.println("I2S initialized.");
  return true;
}

// ============================================================
// Wi-Fi connection
// ============================================================

bool connectWiFi() {
  lcdMessage("WiFi", "Connecting...");
  Serial.print("Connecting WiFi");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - start > 20000) {
      Serial.println("\nWiFi timeout!");
      return false;
    }
  }

  Serial.println();
  Serial.println("WiFi connected.");
  Serial.print("ESP32 IP: "); Serial.println(WiFi.localIP());
  lcdMessageDelay("WiFi", "Connected", 1000);
  return true;
}

// ============================================================
// Setup
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  g_heap_boot = ESP.getFreeHeap();

  Serial.println();
  Serial.println("==========================================");
  Serial.println(" BUMBLEBEE ESP32 (WS2 - FIXED SESSION)");
  Serial.println("==========================================");

  lcd.begin(16, 2);
  lcdMessage("Bumblebee", "Starting...");

  // Heap alloc: pre-roll buffer (250 ms x 16 kHz x 2 B = 8 KB)
  preroll_buffer = (int16_t*)malloc(PREROLL_SAMPLES * sizeof(int16_t));
  if (!preroll_buffer) {
    Serial.println("ERROR: preroll malloc failed.");
    lcdMessage("RAM ERROR", "Preroll alloc");
    while (true) delay(1000);
  }
  memset(preroll_buffer, 0, PREROLL_SAMPLES * sizeof(int16_t));

  // Heap alloc: tensor arena (24 KB)
  tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  if (!tensor_arena) {
    Serial.println("ERROR: tensor_arena malloc failed.");
    lcdMessage("RAM ERROR", "Arena alloc");
    while (true) delay(1000);
  }

  // Measure RAM consumed specifically by Wi-Fi & lwIP network stack
  g_heap_before_wifi = ESP.getFreeHeap();
  if (!connectWiFi()) { lcdMessage("WiFi ERROR", "Check config"); while (true) delay(1000); }
  g_wifi_ram_used = (g_heap_before_wifi > ESP.getFreeHeap()) ? (g_heap_before_wifi - ESP.getFreeHeap()) : 0;

  if (!initializeI2S())      { lcdMessage("I2S ERROR",  "Check wiring"); while (true) delay(1000); }
  if (!initializeFrontend()) { lcdMessage("Frontend",   "ERROR");        while (true) delay(1000); }
  if (!initializeTFLite())   { lcdMessage("TFLite",     "ERROR");        while (true) delay(1000); }

  lcdMessage("Bumblebee", "Listening");
  Serial.println();
  Serial.println("Bumblebee WS2 listening.");
  Serial.print("Free heap at ready: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");
}

// ============================================================
// Main loop
// ============================================================

void loop() {
  int bytes_read = I2S.readBytes((char*)raw_buffer, sizeof(raw_buffer));
  if (bytes_read <= 0) return;

  int samples_read = bytes_read / sizeof(int32_t);

  for (int i = 0; i < samples_read; i++) {
    int64_t v = ((int64_t)raw_buffer[i]) >> 14;
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;
    pcm_buffer[i] = (int16_t)v;
  }

  pushPreroll(pcm_buffer, samples_read);
  processPCM(pcm_buffer, samples_read);
}
