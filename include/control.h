#ifndef __control__ 
#define __control__ 


#include "parameters.h"
#include "field.h"

void enforceNoSlip(struct ComplexField *UK, struct Parameters *params);
void wall_normal_openloop(struct ComplexField *UK, struct Parameters *params, int k_act, double A);
void wall_normal_opposition(struct ComplexField *UK, struct Parameters *params, double G);

#endif