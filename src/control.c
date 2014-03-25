#include <math.h>
#include "field.h"



void enforceNoSlip(struct ComplexField *UK, struct Parameters *params) {
	/* Apply no-slip boundary conditions */

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