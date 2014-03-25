#ifndef __operators_h__
#define __operators_h__

#define PI (2.0*asin(1.0))

#include "parameters.h"
#include "field.h"
#include "fftwplans.h"


void solveVelocityHelmoltzProblems(struct ComplexField *RK, 
				                   struct Parameters *params, 
				                   struct ComplexField *UK);

void solveVorticityStreamFuncHelmoltzProblems(struct ComplexField *RK, 
				                              struct Parameters *params, 
				                              struct ComplexField *UK);

void computeD2DY2(struct ComplexField *UK, struct ComplexField *storage, struct Parameters *params);


void computeRHSHelmoltz(struct ComplexField *UK, 
				        struct ComplexField *UK_old, 
				        struct ComplexField *NK, 
				        struct ComplexField *NK_old, 
				        struct Parameters *params, 
				        struct ComplexField *storage, 
				        struct ComplexField *RK);

void nonLinearTerm(struct ComplexField *UK, 
				   struct RealField *N_store, 
				   struct RealField *Uy_store, 
				   struct RealField *Uz_store, 
				   struct Parameters *params, 
				   struct FFTWPlans *plans, 
				   struct ComplexField *NK);

void complexFieldDifferentiate(struct ComplexField *UK, 
					     	   struct ComplexField *UK_dir, 
					    	   int direction, 
						       struct Parameters *params);

double integralKineticEnergy(struct RealField *U, struct Parameters *params);

void toVelocity(struct ComplexField *UK, struct ComplexField *VK, struct ComplexField *storage_c, struct Parameters *params);

double CFL(struct RealField *U, struct Parameters *params);
#endif