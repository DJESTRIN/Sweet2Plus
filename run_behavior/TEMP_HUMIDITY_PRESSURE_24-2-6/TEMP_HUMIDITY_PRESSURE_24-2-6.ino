

#include <Wire.h>
#include "Adafruit_MPRLS.h"
#include <Adafruit_AHTX0.h>
#include <elapsedMillis.h>

// You dont *need* a reset and EOC pin for most uses, so we set to -1 and don't connect
#define RESET_PIN -1  // set to any GPIO pin # to hard-reset on begin()
#define EOC_PIN -1    // set to any GPIO pin to read end-of-conversion by pin

Adafruit_MPRLS mpr = Adafruit_MPRLS(RESET_PIN, EOC_PIN);
Adafruit_AHTX0 aht;

int waterdelay = 35;
float critscore = 0.8;
int session = 0;
int trial = 0;
int lickedR = 0;
int lickedL = 0;
int lickedNumR = 0;
int lickedNumL = 0;
int realtime = 0;
int ITI = 0;
int totalDispensed = 0; 
int RV;
int RS;
int long currentMillis = 0;
int long startMillis = 0;
//unsigned long startMillis;  //some global variables available anywhere in the program
//unsigned long currentMillis;









void setup() {
  Serial.begin(115200);
  
  


  // Valve Control Pins
  pinMode(2, OUTPUT); //valve 1 Right Water 
  digitalWrite(2,LOW);
  
  pinMode(3, OUTPUT); // Valve 2 Left Water 
  digitalWrite(3,LOW);
  
  pinMode(4, OUTPUT); // Valve 3 Unused Could use a 4 way split to pass air to all other valves 
  digitalWrite(4,LOW);


  pinMode(5, OUTPUT);// Valve 5 (Smell Number 4)
  digitalWrite(5,LOW);

  pinMode(6, OUTPUT);// Valve 6 (Smell Number 3)
  digitalWrite(6,LOW);

  pinMode(7, OUTPUT);// Valve 7 (Smell Number 2)
  digitalWrite(7,LOW);

  
  pinMode(8, OUTPUT);// Valve 8 (Smell Number 1)
  digitalWrite(8,LOW);
  

  pinMode(37, INPUT);// pin that indicates framecout from 2p
  digitalWrite(37,LOW);

  pinMode(18,OUTPUT);// SYNC Pin for triggering behavior camera 
  digitalWrite(18,LOW);

  pinMode(19,OUTPUT);// SYNC Pin for recording arduino, this signal is counted and printed to align frame data on 2p to loop count 
  digitalWrite(19,LOW);

  

  //Touch Sensor
  pinMode(12, INPUT);  // Right Lick Sensor
  pinMode(13, INPUT);  // Left Lick Sensor
  
  //arduino count 
  
  


  //Frame Num Reset Pin
  pinMode(11,INPUT);
  digitalWrite(11,HIGH);
  

  
  //Sensors


  Serial.println("MPRLS Simple Test");
  if (!mpr.begin()) {
  Serial.println("Failed to communicate with MPRLS sensor, check wiring?");
  while (1) {
  delay(10);
  }
  }
  Serial.println("Found MPRLS sensor");

  if (!aht.begin()) {
  Serial.println("Could not find AHT? Check wiring");
  while (1) delay(10);
  }
  Serial.println("AHT10 or AHT20 found");
}


long frameNum = 0000000L;
int i = 0;



void loop() {
  frameNum += 1L;

  if (digitalRead(11) == LOW) {
  frameNum = 0L;
  startMillis = millis();
  }

  //if (totalDispensed <1100){
  //  trial = trial +1;
    
  //}

  //RV = random (5,9);
  //digitalWrite(RV, HIGH);
  //delay (2000);
  //digitalWrite(RV,LOW);
  //Serial.println(RV);
  



  













  int v7 = digitalRead(8);
  int v6 = digitalRead(7);
  int v5 = digitalRead(6);
  int v4 = digitalRead(5);
  //digitalWrite(18,HIGH);
  digitalWrite(19,HIGH);
  delay(1);
  //digitalWrite(18,LOW);
  digitalWrite(19,LOW);
  if (digitalRead(37) == HIGH){
    digitalWrite(18,HIGH);
    delay(1);
    digitalWrite(18,LOW);
  }
  
  currentMillis = (millis()- startMillis);

  float pressure_hPa = mpr.readPressure();
  sensors_event_t humidity, temp;
  aht.getEvent(&humidity, &temp);  // populate temp and humidity objects with fresh data
 Serial.println(String(frameNum) + (",") + (pressure_hPa / 68.947572932) + (",") + (temp.temperature) + (",") + (humidity.relative_humidity) + (",") + (currentMillis) + (",") + (v7) + (",") + (v6) + (",") + (v5) + (",") + (v4));
  //Serial.println(String(frameNum) + (",") + (v7) + (",") + (v6) + (",") + (v5) + (",") + (v4));

  //Serial.println(String(frameNum) + (",") + (v7) + (",") + (v6) + (",") + (v5) + (",") + (v4));

}



void framecount() {

}



