#ifndef __control__ 
#define __control__ 


#include "parameters.h"
#include "field.h"

complex func(struct  ComplexField *UK, struct Parameters *params);
complex integrate_psi_equation(struct  ComplexField *UK, struct Parameters *params);


void applyBC(struct ComplexField *UK, struct Parameters *params, double w0, double psi_upper);
void enforceNoSlip(struct ComplexField *UK, struct Parameters *params);
void wall_normal_openloop(struct ComplexField *UK, struct Parameters *params, int k_act, double A);
void wall_normal_opposition(struct ComplexField *UK, struct Parameters *params, double G);

#endif