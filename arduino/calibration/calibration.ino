#include <MQUnifiedsensor.h>

#define BOARD_NAME          "Adafruit ESP32 Feather"
#define VOLTAGE_RESOLUTION  5
#define ADC_RESOLUTION      12
#define SERIAL_BAUD         115200

#define PIN_MQ3             A3
#define PIN_MQ5             A1
#define PIN_MQ9             A0

#define MQ3_RATIO_CLEAN_AIR     60.0
#define MQ5_RATIO_CLEAN_AIR     6.5
#define MQ9_RATIO_CLEAN_AIR     9.6

#define MQ3_A   4.8387
#define MQ3_B   -2.68
#define MQ5_A   80.897
#define MQ5_B   -2.431
#define MQ9_A   599.65
#define MQ9_B   -2.244

#define WARMUP_MS           300000   // 5 min warmup for sensors
#define SAMPLE_COUNT        100
#define SAMPLE_INTERVAL_MS  100

MQUnifiedsensor MQ3(BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ3, "MQ-3");
MQUnifiedsensor MQ5(BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ5, "MQ-5");
MQUnifiedsensor MQ9(BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ9, "MQ-9");

void initSensor(MQUnifiedsensor &sensor, float a, float b) {
    sensor.setRegressionMethod(1);
    sensor.setA(a); sensor.setB(b);
    sensor.init();
}

float calibrateSensor(MQUnifiedsensor &sensor, float cleanAirRatio) {
    float r0 = 0;
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        sensor.update();
        r0 += sensor.calibrate(cleanAirRatio);
        delay(SAMPLE_INTERVAL_MS);
    }
    return r0 / SAMPLE_COUNT;
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) delay(10);

    initSensor(MQ3, MQ3_A, MQ3_B);
    initSensor(MQ5, MQ5_A, MQ5_B);
    initSensor(MQ9, MQ9_A, MQ9_B);

    Serial.println("# Warming up sensors. Do not expose to any scents");
    for (int remaining = WARMUP_MS / 1000; remaining > 0; remaining -= 10) {
        Serial.print("# ");
        Serial.print(remaining);
        Serial.println("s remaining...");
        delay(10000);
    }

    Serial.println("# Calibrating...");

    float r0_mq3 = calibrateSensor(MQ3, MQ3_RATIO_CLEAN_AIR);
    float r0_mq5 = calibrateSensor(MQ5, MQ5_RATIO_CLEAN_AIR);
    float r0_mq9 = calibrateSensor(MQ9, MQ9_RATIO_CLEAN_AIR);

    Serial.println("\n# Calibration Results");
    Serial.println("# results for main sketch:\n");
    Serial.print("#define MQ3_R0   "); Serial.println(r0_mq3, 4);
    Serial.print("#define MQ5_R0   "); Serial.println(r0_mq5, 4);
    Serial.print("#define MQ9_R0   "); Serial.println(r0_mq9, 4);

    // AI Generated Sanity check for making sure we are reading reasonable values
    Serial.println("\n# Sanity Check");
    if (r0_mq3 < 0.1 || r0_mq3 > 100) Serial.println("# WARNING: MQ3 R0 looks wrong check wiring and 5V power");
    if (r0_mq5 < 0.5 || r0_mq5 > 100) Serial.println("# WARNING: MQ5 R0 looks wrong check wiring and 5V power");
    if (r0_mq9 < 0.5 || r0_mq9 > 100) Serial.println("# WARNING: MQ9 R0 looks wrong check wiring and 5V power");
    Serial.println("\n# Done. Safe to disconnect.");
}

void loop() {}