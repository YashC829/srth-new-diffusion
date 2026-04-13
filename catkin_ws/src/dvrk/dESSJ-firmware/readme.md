# Firmware for dESSJ

The dEESJ board is used in the S/Si SUJs.  These boards provide:
* Pass-through for the serial communication (LVDS) between the dVRK controller and the arm (ESPM)
* SUJ potentiometer sampling using an Arduino board with BlueTooth (BLE) to communicate with the PC

## Build environment

1. Install `arduino-cli`.

    Follow instructions from: https://arduino.github.io/arduino-cli/0.23/installation/

1. plug in the arduino, check if arduino-cli sees it.

    ```sh
    arduino-cli board list
    ```

    Expect something like this

    ```
    Port         Protocol Type              Board Name          FQBN                        Core             
    /dev/ttyACM0 serial   Serial Port (USB) Arduino Nano 33 BLE arduino:mbed_nano:nano33ble arduino:mbed_nano
    ```

1. Install the board and library

     ```sh
     arduino-cli core install arduino:mbed_nano
     arduino-cli lib install ArduinoBLE
     ```

## Compile

`cd` into this directory.

```sh
arduino-cli compile -b arduino:mbed_nano:nano33ble .
```

## Upload the firmware to board

```sh
arduino-cli upload -b arduino:mbed_nano:nano33ble -p /dev/ttyACM0
```

Replace `/dev/ttyACM0` with the serial port that the arduino is on.

If you cannot upload, press the button on the arduino twice. You should see a fading yellow light, indicating the bootloader is running.

## Serial monitor

You can use any serial monitor program to look at the serial output. The baud rate does not matter, because everything is native USB and no actual UART happens.

```sh
arduino-cli monitor -p /dev/ttyACM0
```
