void setup()
{
  Serial.begin(57600);
  //Serial.begin();
}

char accel_mag(char x, char y, char z)
{
  //float sqr_mag = pow(accel[0], 2) + pow(accel[1], 2) + pow(accel[2], 2);
  //return pow(sqr_mag, 0.5);
  return 0;
}

void loop()
{
  // Accelerometer ranges from -512 to 511
//  AccelerationReading reading = Bean.getAcceleration(); 
//  char x = abs(reading.xAxis) / 2;
//  char y = abs(reading.yAxis) / 2;
//  char z = abs(reading.zAxis) / 2;
//  Bean.setLed(x, y, z);
  for(int i = 0; i < 256; i++)
  {
    Bean.setLed(i,i,i);
  }
  Serial.println("Hello there!\n");
  Bean.sleep(1000);
  
//  uint16_t batteryReading = Bean.getBatteryVoltage();
//  String stringToPrint = String();
//  stringToPrint = stringToPrint + "Battery voltage: " + batteryReading/100 + "." + batteryReading%100 + " V";
//  Serial.println(stringToPrint);  
}
