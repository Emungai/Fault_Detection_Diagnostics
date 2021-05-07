#include <conio.h>
#include <iostream>
using namespace std;

const int BITS = 3; // increase this if it doesn't find a solution
const int MASK = (1 << BITS) - 1;

struct OBJECT { int id; char* name; } objects[] = {
  0, "everything else", // default at id 0
  1, "washer segment",
  2, "washer cylinder",
  3, "contact ball",
};
struct NO_COLLISION { int o1, o2; } no_collisions[] = {
  2, 0,
  3, 1,
};
const int NO = _countof(objects);
const int NNC = _countof(no_collisions);
bool forbidden[NO][NO];
unsigned int combi;

void verify_data() {
  for (int i = 0; i < NO; i++) {
    if (objects[i].id != i) throw "?ID SEQUENCE ERROR in objects[]";
  }
  for (int i = 0; i < NNC; i++) {
    if ((unsigned)no_collisions[i].o1 >= NO || (unsigned)no_collisions[i].o2 >= NO) throw "?OUT OF RANGE ERROR in no_collisions[]";
  }
  if (sizeof(combi) * 8 < NO * 2 * BITS) throw "sizeof(combi) too small, use __int64";
}