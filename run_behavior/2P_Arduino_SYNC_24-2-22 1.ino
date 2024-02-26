





void setup() {
  Serial.begin(9600);
  
  
  // Valve Control Pins
  pinMode(20, INPUT);
  pinMode(19, INPUT);
  pinMode(18, INPUT);
  pinMode(6, OUTPUT);
  digitalWrite(6, LOW);
  digitalWrite(20, LOW);
  digitalWrite(19, LOW);
  digitalWrite(18, LOW);

  
  //Frame Trigger Pin 
  attachInterrupt(digitalPinToInterrupt(20),stim, RISING);
  attachInterrupt(digitalPinToInterrupt(19), framecount, RISING);
  attachInterrupt(digitalPinToInterrupt(18),arduino, RISING);
  
  


  //Frame Num Reset Pin
  pinMode(11,INPUT);
  digitalWrite(11,HIGH);
  

  


}

long frameNum = 0L;
//int i = 0;

long behavior = 0L;

long currentbehaviorcount = 0L;
long currentframecount = 0L;


void loop() {  
  currentbehaviorcount = behavior;
  currentframecount = framecount;
  
  if (digitalRead(11) == LOW) {
  frameNum = 0L;
  behavior = 0L;
  }

  if (frameNum >0){ 
  digitalWrite(6,HIGH);
  }
  
  else if (frameNum == 0){
    digitalWrite(6, LOW);
  }

   
}

void arduino(){
 behavior += 1L;
  Serial.println(String(frameNum) + (",") + (behavior)+ (",") + digitalRead(20));


}

void framecount() {
  frameNum +=1L; 
  Serial.println(String(frameNum) + (",") + (behavior) + (",") + digitalRead(20));

}

void stim() {
 


}


