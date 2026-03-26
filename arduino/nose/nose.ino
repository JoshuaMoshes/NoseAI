#include <Wire.h>
#include <Adafruit_BME680.h>
#include <Multichannel_Gas_GMXXX.h>
#include <MQUnifiedsensor.h>

// Hardware Config
#define BOARD_NAME          "Adafruit ESP32 Feather"
#define VOLTAGE_RESOLUTION  5
#define ADC_RESOLUTION      12

#define PIN_MQ9             A0
#define PIN_MQ135           A1
#define PIN_MQ3             A3
#define PIN_MQ9_PREHEAT_5V  3
#define PIN_MQ9_PREHEAT_14V 4

// Pipeline Config
#define SERIAL_BAUD         115200
#define SAMPLE_RATE_MS      200
#define WARMUP_TIME_MS      30000
#define SEA_LEVEL_HPA       1017.0

// Calibration
#define MQ135_R0            14.29
#define MQ9_R0              2.96
#define MQ3_R0              0.04

// MQ Sensor Curve Coefficients (from datasheets)
// PPM = A * (Rs/R0)^B  (regression method 1 = power law)
// Target gas per sensor chosen to match SmellNET paper

#define MQ135_A     77.255   // Alcohol
#define MQ135_B     -3.18

#define MQ9_A       599.65   // LPG
#define MQ9_B       -2.244

#define MQ3_A       4.8387   // Benzene
#define MQ3_B       -2.68

#define POWER_LAW_REGRESSION  1  // MQUnifiedsensor regression method

// Clean Air Ratios (Rs/R0 in clean air, from datasheets)
#define MQ135_RATIO_CLEAN_AIR   3.6
#define MQ9_RATIO_CLEAN_AIR     9.6
#define MQ3_RATIO_CLEAN_AIR     60.0

// Sensor Objects
Adafruit_BME680      bme;
GAS_GMXXX<TwoWire>   gas;
MQUnifiedsensor      MQ135(BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ135, "MQ-135");
MQUnifiedsensor      MQ9  (BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ9,   "MQ-9");
MQUnifiedsensor      MQ3  (BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ3,   "MQ-3");

uint32_t sampleIndex = 0;

void initBME680() {
    if (!bme.begin()) {
        Serial.println("{\"error\":\"BME680 not found\"}");
        while (true) delay(1000);
    }
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150);
}

void initMQSensors() {
    pinMode(PIN_MQ9_PREHEAT_5V,  OUTPUT);
    pinMode(PIN_MQ9_PREHEAT_14V, OUTPUT);

    MQ135.setRegressionMethod(POWER_LAW_REGRESSION);
    MQ135.setA(MQ135_A); MQ135.setB(MQ135_B);
    MQ135.init(); MQ135.setR0(MQ135_R0);

    MQ9.setRegressionMethod(POWER_LAW_REGRESSION);
    MQ9.setA(MQ9_A); MQ9.setB(MQ9_B);
    MQ9.init(); MQ9.setR0(MQ9_R0);

    MQ3.setRegressionMethod(POWER_LAW_REGRESSION);
    MQ3.setA(MQ3_A); MQ3.setB(MQ3_B);
    MQ3.init(); MQ3.setR0(MQ3_R0);
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) delay(10);

    Wire.begin();
    initBME680();

    #ifndef BME680_ONLY
        gas.begin(Wire, 0x08);
        initMQSensors();
    #endif

    Serial.println("{\"status\":\"warming_up\"}");
    delay(WARMUP_TIME_MS);
    Serial.println("{\"status\":\"ready\"}");
}

void jsonFloat(const char* key, float val, int decimals, bool last = false) {
    Serial.print('"'); Serial.print(key); Serial.print("\":");
    Serial.print(val, decimals);
    if (!last) Serial.print(',');
}

void jsonUint(const char* key, uint32_t val, bool last = false) {
    Serial.print('"'); Serial.print(key); Serial.print("\":");
    Serial.print(val);
    if (!last) Serial.print(',');
}

void emitReading() {
    if (!bme.performReading()) {
        Serial.println("{\"error\":\"BME680 read failed\"}");
        return;
    }

    Serial.print('{');
    jsonUint ("idx", sampleIndex++,             false);
    jsonUint ("ts",  millis(),                   false);

    #ifndef BME680_ONLY
        jsonFloat("no2",    (float)gas.measure_NO2(),    0, false);
        jsonFloat("c2h5oh", (float)gas.measure_C2H5OH(), 0, false);
        jsonFloat("voc",    (float)gas.measure_VOC(),    0, false);
        jsonFloat("co",     (float)gas.measure_CO(),     0, false);
        MQ135.update(); MQ9.update(); MQ3.update();
        jsonFloat("mq135",  MQ135.readSensor(),          2, false);
        jsonFloat("mq9",    MQ9.readSensor(),             2, false);
        jsonFloat("mq3",    MQ3.readSensor(),             2, false);
    #endif

    jsonFloat("temp",     bme.temperature,             2, false);
    jsonFloat("pressure", bme.pressure / 100.0,        2, false);
    jsonFloat("humidity", bme.humidity,                2, false);
    jsonFloat("gas_res",  bme.gas_resistance / 1000.0, 2, false);
    jsonFloat("altitude", bme.readAltitude(SEA_LEVEL_HPA), 2, true);
    Serial.println('}');
}

void loop() {
    uint32_t cycleStart = millis();

    emitReading();

    uint32_t elapsed = millis() - cycleStart;
    if (elapsed < SAMPLE_RATE_MS) delay(SAMPLE_RATE_MS - elapsed);
}