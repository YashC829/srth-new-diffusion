#include <ArduinoBLE.h>
#include <queue>

BLEService sujservice("babb122c-de4c-11ec-9d64-0242ac120100");

const size_t nb_axis = 8;

// moving average variables
typedef std::array<uint32_t, nb_axis> sample_type;
std::queue<sample_type> queue;
const size_t window_size = 16;
sample_type samples = {0};
sample_type sums = {0};
uint16_t averages[nb_axis] = {0};

BLEStringCharacteristic ble_pots_characteristic("babb122c-de4c-11ec-9d64-0242ac120101",  // standard 16-bit characteristic UUID
						BLERead | BLENotify, 128); // remote clients will be able to get notifications if this characteristic changes

// mapping between pots and A2D pins on Arduino
uint8_t analog_pins[nb_axis] = {A7, A5, A3, A1, A6, A4, A2, A0};
char output_string[256] = "";

void setup() {
  Serial.begin(9600);
  analogReadResolution(12);

  pinMode(LED_BUILTIN, OUTPUT);
  for (int i = 0; i < nb_axis; i++) {
    pinMode(analog_pins[i], INPUT);
  }

  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  BLE.setDeviceName("dVRK SUJ");
  BLE.setLocalName("dVRK SUJ");

  sujservice.addCharacteristic(ble_pots_characteristic);
  BLE.addService(sujservice);
  BLE.setAdvertisedService(sujservice);

  BLE.advertise();

  Serial.println("Bluetooth device active, waiting for connections...");
}

void loop() {

  BLE.poll();
  // read 16 times to get a 16 bit output out of 12 bits A2D.  This
  // assumes samples[i] is set to zero
  for (size_t nb_samples = 0; nb_samples < 16; ++nb_samples) {
    for (size_t index = 0; index < nb_axis; ++index) {
      samples[index] += analogRead(analog_pins[index]);
    }
  }

  // for each axis, compute moving average
  const size_t queue_size = queue.size();
  if (queue_size == window_size) {
    for (size_t index = 0; index < nb_axis; ++index) {
      sums[index] -= (queue.front())[index];
    }
    queue.pop();
  }

  // in any case, push the new values
  queue.push(samples);
  // and compute moving average
  for (size_t index = 0; index < nb_axis; ++index) {
    sums[index] += samples[index];
    averages[index] = sums[index] / window_size;
    samples[index] = 0;
  }

  sprintf(output_string, "{\"timestamp\":%U, \"pots\":[[%u,%u,%u,%u],[%u,%u,%u,%u]]}", millis(), averages[0], averages[1], averages[2], averages[3], averages[4], averages[5], averages[6], averages[7]);
  ble_pots_characteristic.writeValue(output_string);
}
