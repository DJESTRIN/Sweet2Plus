#include <random>
#include <iostream>
using namespace std;

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

int main(){
    int numberoh;
    numberoh = random(100,200) ;
    cout << "numberoh is equal to " << numberoh; 
    return 0;

    }