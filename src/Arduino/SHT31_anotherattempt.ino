
#include "Adafruit_SHT31.h"

Adafruit_SHT31 sht31 = Adafruit_SHT31();

void setup() {
  
  sht31.begin(0x44);
  Serial.begin(9600);
}

void loop() {
  float t = sht31.readTemperature();
  float h = sht31.readHumidity();

  Serial.print("Temp *C = ");
  Serial.println(t);

  Serial.print("Hum. % = ");
  Serial.println(h);

  Serial.println();
  Bean.sleep(1000);

}
