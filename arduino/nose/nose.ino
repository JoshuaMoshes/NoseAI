#include <Wire.h>
#include <Adafruit_BME680.h>
#include <Multichannel_Gas_GMXXX.h>
#include <MQUnifiedsensor.h>

// Hardware Config
#define BOARD_NAME          "Adafruit ESP32 Feather"
#define VOLTAGE_RESOLUTION  5
#define ADC_RESOLUTION      12

#define PIN_MQ3             A3   // ADC1
#define PIN_WSP2110         A2   // ADC1
#define PIN_MP503           A4   // ADC1
#define PIN_MQ9             A0   // ADC2
#define PIN_MQ5             A1   // ADC2

#define PIN_MQ9_PREHEAT_5V  3
#define PIN_MQ9_PREHEAT_14V 4

// Pipeline Config
#define SERIAL_BAUD         115200
#define SAMPLE_RATE_MS      200
#define WARMUP_TIME_MS      30000
#define SEA_LEVEL_HPA       1017.0

// Calibration
#define MQ3_R0              3.6226
#define MQ5_R0              41.8166
#define MQ9_R0              25.3214
#define WSP2110_R0          10.1   // from SmellNET paper

// MQ Curve Coefficients
// MQ-3 Benzene from datasheet
#define MQ3_A               4.8387
#define MQ3_B               -2.68

// MQ-5 LPG from datasheet
#define MQ5_A               80.897
#define MQ5_B               -2.431

// MQ-9 CO from datasheet
#define MQ9_A               599.65
#define MQ9_B               -2.244

#define POWER_LAW_REGRESSION    1

// Clean Air Ratios
#define MQ3_RATIO_CLEAN_AIR     60.0
#define MQ5_RATIO_CLEAN_AIR     6.5
#define MQ9_RATIO_CLEAN_AIR     9.6

// WSP2110 (HCHO) formula constants from SmellNET
// ppm = 10 ^ ((log10(Rs/R0) - 0.0827) / -0.4807)
#define WSP2110_LOG_OFFSET      0.0827
#define WSP2110_LOG_SLOPE       -0.4807

// Sensor Objects
Adafruit_BME680      bme;
GAS_GMXXX<TwoWire>   gas;
MQUnifiedsensor      MQ3 (BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ3,  "MQ-3");
MQUnifiedsensor      MQ5 (BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ5,  "MQ-5");
MQUnifiedsensor      MQ9 (BOARD_NAME, VOLTAGE_RESOLUTION, ADC_RESOLUTION, PIN_MQ9,  "MQ-9");

uint32_t sampleIndex = 0;

float readWSP2110() {
    int raw = analogRead(PIN_WSP2110);
    if (raw <= 0) return 0.0;
    float rs = (4095.0 / raw) - 1.0;
    return pow(10.0, (log10(rs / WSP2110_R0) - WSP2110_LOG_OFFSET) / WSP2110_LOG_SLOPE);
}

float readMP503() {
    int raw = analogRead(PIN_MP503);
    if (raw <= 0) return 0.0;
    return (raw / 4095.0) * VOLTAGE_RESOLUTION; // raw voltage
}

// Init

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

    MQ3.setRegressionMethod(POWER_LAW_REGRESSION);
    MQ3.setA(MQ3_A); MQ3.setB(MQ3_B);
    MQ3.init(); MQ3.setR0(MQ3_R0);

    MQ5.setRegressionMethod(POWER_LAW_REGRESSION);
    MQ5.setA(MQ5_A); MQ5.setB(MQ5_B);
    MQ5.init(); MQ5.setR0(MQ5_R0);

    MQ9.setRegressionMethod(POWER_LAW_REGRESSION);
    MQ9.setA(MQ9_A); MQ9.setB(MQ9_B);
    MQ9.init(); MQ9.setR0(MQ9_R0);
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
    MQ3.update(); MQ5.update(); MQ9.update();

    if (!bme.performReading()) {
        Serial.println("{\"error\":\"BME680 read failed\"}");
        return;
    }

    Serial.print('{');
    jsonUint ("idx",          sampleIndex++,                       false);
    jsonUint ("ts",           millis(),                             false);

    // Grove Multichannel V2
    jsonFloat("NO2",          (float)gas.measure_NO2(),        0,  false);
    jsonFloat("C2H5OH",       (float)gas.measure_C2H5OH(),     0,  false);
    jsonFloat("VOC",          (float)gas.measure_VOC(),        0,  false);
    jsonFloat("CO",           (float)gas.measure_CO(),         0,  false);

    // Analog sensors
    jsonFloat("Alcohol",      readWSP2110(),                    2,  false);
    jsonFloat("LPG",          MQ9.readSensor(),                 2,  false);
    jsonFloat("Benzene",      MQ3.readSensor(),                 2,  false);
    jsonFloat("MQ5",          MQ5.readSensor(),                 2,  false); // extra, the paper does not actually have this column
    jsonFloat("MP503",        readMP503(),                      3,  false); // also extra, the paper did not collect this

    // BME680
    jsonFloat("Temperature",  bme.temperature,                  2,  false);
    jsonFloat("Pressure",     bme.pressure / 100.0,             2,  false);
    jsonFloat("Humidity",     bme.humidity,                     2,  false);
    jsonFloat("Gas_Resistance", bme.gas_resistance / 1000.0,   2,  false);
    jsonFloat("Altitude",     bme.readAltitude(SEA_LEVEL_HPA),  2,  true);
    Serial.println('}');
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) delay(10);

    Wire.begin();
    gas.begin(Wire, 0x08);
    initBME680();
    initMQSensors();

    Serial.println("{\"status\":\"warming_up\"}");
    delay(WARMUP_TIME_MS);
    Serial.println("{\"status\":\"ready\"}");
}

void loop() {
    uint32_t cycleStart = millis();
    emitReading();
    uint32_t elapsed = millis() - cycleStart;
    if (elapsed < SAMPLE_RATE_MS) delay(SAMPLE_RATE_MS - elapsed);
}