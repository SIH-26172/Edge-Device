# Bumblebee ESP32 Memory Test

ESP32 Arduino firmware that detects the Bumblebee wake word with an embedded TensorFlow Lite Micro model, displays status on a 16x2 LCD, and captures microphone audio over I2S.

## Hardware

- ESP32 development board
- I2S microphone
- 16x2 parallel LCD

### Wiring

| Device | ESP32 pin |
| --- | ---: |
| I2S BCLK | 18 |
| I2S WS/LRCLK | 19 |
| I2S DIN | 23 |
| LCD RS | 21 |
| LCD E | 22 |
| LCD D4 | 4 |
| LCD D5 | 5 |
| LCD D6 | 33 |
| LCD D7 | 2 |

Use the correct power and ground connections for the specific microphone and LCD module.

## Arduino IDE setup

1. Install the ESP32 board package and select the matching ESP32 board.
2. Copy `config.h.example` to `config.h`.
3. Edit `config.h` with the Wi-Fi name, Wi-Fi password, and the IP address required by the sketch before uploading.
4. Install or copy these libraries into the Arduino libraries folder:
   - `Chirale_TensorFLite`
   - `LiquidCrystal`
   - The ESP32 board package provides `WiFi.h` and `ESP_I2S.h`.
5. Open `BumblebeeMemoryTest1.ino` in Arduino IDE.
6. Select the board and port, then compile and upload.
7. Open Serial Monitor at `115200` baud.

The local `config.h` is ignored by Git because it contains credentials. Do not commit it.

## Repository layout

```text
BumblebeeMemoryTest1/
  BumblebeeMemoryTest1.ino
  config.h.example
  model.h
  bumblebee_working_v1.tflite
```

## GitHub upload

This project should be uploaded inside the GitHub repository folder you choose, for example:

```text
your-repository/
  projects/
    BumblebeeMemoryTest1/
```

Run the Git commands from the GitHub repository root. Replace `projects/BumblebeeMemoryTest1` with the exact folder path you want.

```powershell
git init
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git add projects/BumblebeeMemoryTest1
git commit -m "Add Bumblebee ESP32 firmware"
git push -u origin main
```

If the GitHub repository already contains files, clone it first and copy this project into the desired subfolder instead of running `git init` inside this project folder.