#include <math.h>
#include <complex.h>
#include "field.h"
#include "output.h"

void enforceNoSlip(struct ComplexField *UK, struct Parameters *params) {
	// Apply no-slip boundary conditions. This will set the boundary
	// conditions on u, psi and omega, as if we had no motion of the 
	// wall. 

	for (int k=0; k<UK->Nz/2+1; k++) {

		// condition on u
		index3dC(UK, 0, 0, k) = 0.0 + I*0.0;
		index3dC(UK, 0, UK->Ny-1, k) = 0.0 + I*0.0;

		// condition on v, comes from psi
		index3dC(UK, 2, 0, k) = 0.0 + I*0.0;
		index3dC(UK, 2, UK->Ny-1, k) = 0.0 + I*0.0;

		// condition on w, comes from omega equation and condition on psi
		index3dC(UK, 1, 0, k) = -2.0*index3dC(UK, 2, 1, k)/pow(params->h[0], 2);
		index3dC(UK, 1, UK->Ny-1, k) = -2.0*index3dC(UK, 2, UK->Ny-2, k)/pow(params->h[UK->Ny-2], 2);
	}
}

void applyBC(struct ComplexField *UK, struct Parameters *params, double w0, double psi_upper) {
	// This function will make sure that we have the correct boundary
	// conditions applied on both the upper and lower walls. It will
	// do the following things:
	// 1) set the value of the streamfunction on the upper wall, depending
	//    on the type of condition that we want to match, i.e. zero pressure
	//	  gradient or zero net mass flux. This will only set the value for
	//    the "mean" term, i.e. for the Fourier mode for k=0, since the
	//    others are strictly zero. On the lower wall the streamfunction
	// 	  is always zero. 
	// 2) set u to zero on the walls.
	// 3) set the vorticity value on the lower wall, depending on the
	// 	  value of the wall velocity.

	// IT IS NOT IMPLEMENTED FOR GRID STRETCHING, (IS IT?)

	// set zeros first, which 
	enforceNoSlip(UK, params);

	// boundary condition for omega
	index3dC(UK, 1, 0, 0) -= 2/params->h[0]*w0*UK->Nz;
	index3dC(UK, 1, UK->Ny-1, 0) += 2.0*psi_upper*UK->Nz/pow(params->h[UK->Ny-2], 2);

	// boudnary condition for psi
	index3dC(UK, 2, UK->Ny-1, 0) = psi_upper*UK->Nz;
}

void wall_normal_openloop(struct ComplexField *UK, struct Parameters *params, int k_act, double A) {
	/*	Set v(z, -h, t) = A*sin(2*PI/L*k_act*z) on bottom wall only.
	*/

	double h = 2.0/(UK->Ny - 1);

	// first enforce not slip, so we have zeros where we do not need them.
	enforceNoSlip(UK, params);

	// apply condition on psi, for the wall normal component
	index3dC(UK, 2, 0, k_act) = A/(I*k_act*params->alpha);

	// apply condition on omega for the axial component
	index3dC(UK, 1, 0, k_act) =  k_act*k_act*params->alpha*params->alpha*index3dC(UK, 2, 0, k_act)
					           - 2.0*(index3dC(UK, 2, 1, k_act) - index3dC(UK, 2, 0, k_act))/(h*h);
}

void wall_normal_opposition(struct ComplexField *UK, struct Parameters *params, double G) {
	/*	Set v(z, -h, t) = A*sin(2*PI/L*k_act*z) on bottom wall only.
	*/

	double h = 2.0/(UK->Ny - 1);

	// first enforce not slip, so we have zeros where we do not need them.
	enforceNoSlip(UK, params);

	for (int k=1; k<UK->Nz/2+1; k++) {
		// apply condition on psi, for the wall normal component
		index3dC(UK, 2, 0, k) = G*(index3dC(UK, 0, UK->Ny-2, k))/(I*k*params->alpha);

		// apply condition on omega for the axial component to be zero
		index3dC(UK, 1, 0, k) =  k*k*params->alpha*params->alpha*index3dC(UK, 2, 0, k)
					           	- 2.0*(index3dC(UK, 2, 1, k) - index3dC(UK, 2, 0, k))/(h*h);	
    }
}