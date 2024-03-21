#include <time.h> 
void setup() {
  pinMode(8, OUTPUT); // valve 1 Vanilla
  digitalWrite(8,LOW);
  
  pinMode(7, OUTPUT); // valve 2 Penut Oil
  digitalWrite(7,LOW);

  pinMode(6, OUTPUT); // valve 3 Water
  digitalWrite(6,LOW);

  pinMode(5, OUTPUT); // valve 4 TMT
  digitalWrite(5,LOW);


  pinMode(3, INPUT); // Forces it high so we can hit button forces low and experiment starts
  digitalWrite(3,HIGH);

  int counter;
  counter=0;
  int step;
  step=1;
}

int random(int min, int max) //range : [min, max]
{
   static bool first = true;
   if (first) 
   {  
      srand( time(NULL) ); //seeding for the first time only!
      first = false;
   }
   return min + rand() % (( max + 1 ) - min);
}

void loop() {
  if (digitalRead (3) == LOW){
    // Wait 15 minutes
      delay(900000);

    // Run randomly the vanilla, water or peanut oil smells for x trials
    for (int x = 0; x < 60; x++) {
      int randomvalve;
      randomvalve = random(6,8);
      delay(15000);
      digitalWrite(randomvalve, HIGH);
      delay(5000);
      digitalWrite(randomvalve,LOW);
      delay(15000);
    }

    // Run TMT trials
    for (int x = 0; x < 5; x++) {
          delay(15000);
          digitalWrite(5, HIGH);
          delay(5000);
          digitalWrite(5,LOW);
          delay(15000);
        }

      

  }
}
