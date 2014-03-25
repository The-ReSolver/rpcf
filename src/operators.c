#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#define MKL_Complex16 complex
#include <mkl.h>
//#include <lapacke.h>
#include <omp.h>


#include "field.h"
#include "dbg.h"
#include "operators.h"
#include "fftwplans.h"
#include "solver.h"

// macro to index over the matrix arising from the solution of the helmoltz problems for psi and omega
#define index_matrix_lapack(vals, i, j) (vals)[(4 - (j) + (i)) + (j)*7]


void solveVelocityHelmoltzProblems(struct ComplexField *RK, struct Parameters *params, struct ComplexField *UK) {

	// first allocate pointers
	complex *lower_diag;
	complex *upper_diag;
	complex *main_diag;
	complex *x;
	
	int info;

	// coefficients for the derivative
	double *ai = malloc(sizeof(double)*(RK->Ny-2));
	double *bi = malloc(sizeof(double)*(RK->Ny-2));
	double *ci = malloc(sizeof(double)*(RK->Ny-2));

	// precompute coefficients once for all
	for (int j=1; j<params->Ny-1; j++) {
		ai[j-1] =   2.0/(params->h[j-1]*(params->h[j-1] + params->h[j]));
		bi[j-1] = - 2.0/(params->h[j-1]* params->h[j]);
		ci[j-1] =   2.0/(params->h[j]  *(params->h[j-1] + params->h[j]));
	}

	# pragma omp parallel default(shared) private(lower_diag, upper_diag, main_diag, x, info)
	{

		// each thread allocates its stuff
		lower_diag = malloc(sizeof(complex)*(params->Ny-2));
		upper_diag = malloc(sizeof(complex)*(params->Ny-2));
		main_diag  = malloc(sizeof(complex)*(params->Ny-2));
		x          = malloc(sizeof(complex)*(params->Ny-2));

		#pragma omp for
		for (int k=0; k<params->Nz/2+1; k++) {
			for (int j=1; j<params->Ny-1; j++) {
				 main_diag[j-1] = bi[j-1] - k*k*params->alpha*params->alpha - 2.0*params->Re/params->dt;
				lower_diag[j-1] = ai[j-1];
				upper_diag[j-1] = ci[j-1];
			}
			
			for (int j=1; j<params->Ny-1; j++) {
				x[j-1] = index3dC(RK, 0, j, k);
			}

			info = LAPACKE_zgtsv(LAPACK_COL_MAJOR, params->Ny-2, 1, &lower_diag[1], main_diag, upper_diag, x, params->Ny-2);

			if (info != 0) {
				log_err("Info flag equal to %d", info);
				exit(1);
			}
			
			for (int j=1; j<params->Ny-1; j++) {
				index3dC(UK, 0, j, k) = x[j-1];
			}
		}

		// free memory
		free(lower_diag);
		free(upper_diag);
		free(main_diag);
		free(x);
	}
	free(ai);
	free(bi);
	free(ci);
}

void solveVorticityStreamFuncHelmoltzProblems(struct ComplexField *RK, struct Parameters *params, struct ComplexField *UK) {
	/*	Solve coupled Hlemholtz problem for vorticity and streamfunction.
	*/

	// lapack solutions things
	int M = 2*(params->Ny - 2);  // size of the matrix
	int ldab = 7;                // Leading dimension of the matrix. 2*kl + ku + 1
	long long ipiv[M];                 // Leading dimension of the matrix. 2*kl + ku + 1
	int info;

	// working areas for solution of system
	complex *vals;
	complex *b; // this contains the known terms and will contain the solution

	# pragma omp parallel default(shared) private(vals, b, ipiv, info)
	{

		// array to store the diagonals
		vals = malloc(sizeof(complex)*ldab*M);
	
		// array to store the vector of known terms + solution
		b = malloc(sizeof(complex)*M);
	

		#pragma omp for
		for (int k=0; k<params->Nz/2+1; k++) {

			// fill vector of known terms
			for (int j=1; j<params->Ny-1; j++) {
				b[2*(j-1)] = index3dC(RK, 1, j, k); // 0, 2, 4, ...
				b[2*j - 1] = 0.0;                   // 1, 3, 5, ...
			}

			for (int jj=0; jj<ldab*M; jj++) {
				vals[jj] = 0.0;
			}

			// now fill diagonals
			// main diagonal
			for (int j=1; j<params->Ny-1; j++) {
				// the value is different for psi and omega, and we 
				// use a j based indexing 
				index_matrix_lapack(vals, 2*(j-1), 2*(j-1)) = - 2.0/(params->h[j]*params->h[j-1]) 
												              - pow(k*params->alpha, 2) 
												              - 2.0*params->Re/params->dt;
				index_matrix_lapack(vals, 2*j-1, 2*j-1)     = - 2.0/(params->h[j]*params->h[j-1]) 
												              - pow(k*params->alpha, 2);
			}
			
			// diagonal +2
			for (int j=1; j<params->Ny-2; j++) {
				index_matrix_lapack(vals, 2*(j-1), 2*(j-1) + 2) = 2.0/(params->h[j]*(params->h[j] + params->h[j-1]));
				index_matrix_lapack(vals, 2*j-1,   2*j-1   + 2) = 2.0/(params->h[j]*(params->h[j] + params->h[j-1]));
			}

			// diagonal -2
			for (int j=2; j<params->Ny-1; j++) {
				index_matrix_lapack(vals, 2*(j-1), 2*(j-1) - 2) = 2.0/(params->h[j-1]*(params->h[j] + params->h[j-1]));
				index_matrix_lapack(vals, 2*j-1,   2*j-1   - 2) = 2.0/(params->h[j-1]*(params->h[j] + params->h[j-1]));
			}

			// diagonal +1
			for (int i=0; i<M-1; i++) {
				index_matrix_lapack(vals, i, i+1) = 0.0;
			}
			
			// add boundary conditions
			// this results from the condition on w, where we impose the 
			// value of omega at the wall to satisfy the streamfunction
			// definition.
			index_matrix_lapack(vals, 0, 1)     = - 2.0/pow(params->h[0], 2) 
												  * 2.0/(params->h[0]*(params->h[1] + params->h[0]));


			index_matrix_lapack(vals, M-2, M-1) = - 2.0/pow(params->h[params->Ny-2], 2) 
												  *	2.0/(params->h[params->Ny-2]*(params->h[params->Ny-3] + params->h[params->Ny-2]));

			// this is when i have v != 0 on the bottom boundary
			// b[0] -= index3dC(UK, 2, 0, k)*(k*k*params->alpha*params->alpha/(h*h) + 2.0/(h*h*h*h));
			// b[1] -= index3dC(UK, 2, 0, k)/(h*h);

			// diagonal -1
			for (int i=1; i<M; i++) {
				if (i%2 == 1) {
					index_matrix_lapack(vals, i, i-1) = 1.0;
				} else {
					index_matrix_lapack(vals, i, i-1) = 0.0;
				}
			}

	    	// solve system
			info = LAPACKE_zgbsv(LAPACK_COL_MAJOR, M, 2, 2, 1, vals, ldab, ipiv, b, M);
			if (info != 0) {
				log_err("Info flag equal to %d", info);
				exit(1);
			}

			// now copy back to solution vector
			for (int j=1; j<params->Ny-1; j++) {
				index3dC(UK, 1, j, k) = b[2*(j-1)]; // 0, 2, 4, ...
				index3dC(UK, 2, j, k) = b[2*j-1];   // 1, 3, 5, ...
			}
		}
	
		// free memory
		free(vals);
		free(b);
	}
}

void computeD2DY2(struct ComplexField *UK, struct ComplexField *storage, struct Parameters *params){
	/* 	Compute second derivative along y of solution variables.

		We assume boundary conditions have been enforced at this stage,
		because we use the upper and lower grid points for the computations.
	*/
	double ai, bi, ci;

	for (int i=0; i<3; i++) {
		for (int j=1; j<UK->Ny-1; j++) {
			ai =   2.0/(params->h[j-1]*(params->h[j-1] + params->h[j]));
			bi = - 2.0/(params->h[j-1]* params->h[j]);
			ci =   2.0/(params->h[j]  *(params->h[j-1] + params->h[j]));
			for (int k=0; k<UK->Nz/2+1; k++) {
				index3dC(storage, i, j, k) =   ai*index3dC(UK, i, j-1, k) 
						  				 	 + bi*index3dC(UK, i, j,   k) 
										 	 + ci*index3dC(UK, i, j+1, k); 
			}
		}
	}

	// update exterior points, by copying results from the innermost point.
	// this is probably not needed
	for (int i=0; i<3; i++) {
		for (int k=0; k<UK->Nz/2+1; k++) {
			index3dC(storage, i, 0, k) = index3dC(storage, i, 1, k);
			index3dC(storage, i, UK->Ny-1, k) = index3dC(storage, i, UK->Ny-2, k);
		}
	}
}

void computeRHSHelmoltz(struct ComplexField *UK, struct ComplexField *UK_old, struct ComplexField *NK, struct ComplexField *NK_old, struct Parameters *params, struct ComplexField *storage, struct ComplexField *RK) {
	/* 	Compute right-hand-side of helmoltz problems for the velocity
	  	and vorticity equations. For the stream function equation this
		step is done in the routine solveHelmoltzProblems.
	*/

	// first need to compute second derivatives
	computeD2DY2(UK, storage, params);
	int kk;

	# pragma omp parallel for default(shared) private(kk)
	for (int j=1; j<params->Ny-1; j++) {
		for (int k=0; k<params->Nz/2+1; k++) {
			kk = (k != UK->Nz/2) ? k : -k;
			index3dC(RK, 0, j, k) = + (k*k*params->alpha*params->alpha - 2*params->Re/params->dt)*index3dC(UK, 0, j, k)
			                        - index3dC(storage, 0, j, k)
			                        - params->Re*(params->Ro-1)*I*kk*params->alpha*(3*index3dC(UK, 2, j, k) - index3dC(UK_old, 2, j, k))
			                        - params->Re*(3*index3dC(NK, 0, j, k) - index3dC(NK_old, 0, j, k));
			index3dC(RK, 1, j, k) = + (k*k*params->alpha*params->alpha - 2*params->Re/params->dt)*index3dC(UK, 1, j, k)
			                        - index3dC(storage, 1, j, k)
			                        - params->Re*params->Ro*I*kk*params->alpha*(3*index3dC(UK, 0, j, k) - index3dC(UK_old, 0, j, k))
			                        - params->Re*(3*index3dC(NK, 1, j, k) - index3dC(NK_old, 1, j, k));
		}
	}
}

void nonLinearTerm(struct ComplexField *UK, struct RealField *N_store, struct RealField *Uy_store, struct RealField *Uz_store, struct Parameters *params, struct FFTWPlans *plans, struct ComplexField *NK){

	// NOTE:  use NK as temporary storage for derivatives

	// now differentiate along z
	complexFieldDifferentiate(UK, NK, 0, params);	
	ifft(NK, Uz_store, plans);

	// now differentiate along y
	complexFieldDifferentiate(UK, NK, 1, params);	
	ifft(NK, Uy_store, plans);

	// multiply in physical space
	// Note that we also compute stuff at the boundaries, but this should be considered as garbage
	# pragma omp parallel for default(shared)
	for(int j=0; j<params->Ny*params->Nz; j++){
		index(N_store, 0, j) = -index(Uz_store, 2, j)*index(Uy_store, 0, j) + index(Uy_store, 2, j)*index(Uz_store, 0, j);
		index(N_store, 1, j) = -index(Uz_store, 2, j)*index(Uy_store, 1, j) + index(Uy_store, 2, j)*index(Uz_store, 1, j);
	}

	// now get non linear term in fourier space
	fft(N_store, NK, plans);
}

void complexFieldDifferentiate(struct ComplexField *UK, struct ComplexField *UK_dir, int direction, struct Parameters *params){
	/* Differentiate UK with respect to some direction and write into UK_dir.

	Parameters
	----------
	dir: the direction: 0 for z, 1 for y

	*/
	
	int kk; // true wave number
	double ai, bi, ci; // coefficients of the first order derivative

	if(direction == 0) { // along z
		for (int i=0; i<3; i++){
			for (int j=1; j<UK->Ny-1; j++){
				for (int k=0; k<UK->Nz/2+1; k++){
					kk = (k != UK->Nz/2) ? k : -k;
					index3dC(UK_dir, i, j, k) = I*kk*params->alpha*index3dC(UK, i, j, k);
				}
			}
		}
	} else if (direction == 1) { // along y
		for (int i=0; i<3; i++){
			for (int j=1; j<UK->Ny-1; j++){

				ai = -params->h[j]   / params->h[j-1]  / (params->h[j-1] + params->h[j]);
				bi = (params->h[j]   - params->h[j-1]) / (params->h[j-1] * params->h[j]);
				ci =  params->h[j-1] / params->h[j]    / (params->h[j-1] + params->h[j]);

				for (int k=0; k<UK->Nz/2+1; k++){
					index3dC(UK_dir, i, j, k) =  ai*index3dC(UK, i, j-1, k) 
						  				 	   + bi*index3dC(UK, i, j,   k) 
										 	   + ci*index3dC(UK, i, j+1, k); 
				}
			}
		}
	}
}

void toVelocity(struct ComplexField *UK, struct ComplexField *VK, struct ComplexField *storage_c, struct Parameters *params) {
	/*	Get velocity components from streamfunction. We need one storage for computation of the 
		derivative.
	*/
	// copy u 
	memcpy(VK->component[0], UK->component[0], sizeof(fftw_complex)*UK->Ny*(UK->Nz/2+1));

	// derive along z for v
	complexFieldDifferentiate(UK, storage_c, 0, params);
	memcpy(VK->component[1], storage_c->component[2], sizeof(fftw_complex)*UK->Ny*(UK->Nz/2+1));

	// derive along y for w
	complexFieldDifferentiate(UK, storage_c, 1, params);
	// change sign cause : w = - dpsi/dy
	for (int j=0; j<UK->Ny*(UK->Nz/2+1); j++){
		index(storage_c, 2, j) *= -1.0;
	}
	memcpy(VK->component[2], storage_c->component[2], sizeof(fftw_complex)*UK->Ny*(UK->Nz/2+1));

	// set to zero the bc
	for (int k=0; k<UK->Nz/2+1; k++) { 
		index3dC(VK, 1, 0, k) = 0.0;
		index3dC(VK, 2, 0, k) = 0.0;
		index3dC(VK, 1, UK->Ny-1, k) = 0.0;
		index3dC(VK, 2, UK->Ny-1, k) = 0.0;
	}
}

double integralKineticEnergy(struct RealField *U, struct Parameters *params) {
	/* 	Compute integral of the kinetic energy over the domain
		with trapezoidal rule in physical space.
	*/
	double sum = 0.0;
	
	for (int c=0; c<3; c++) { 
		// inner points
		for (int j=1; j<U->Ny-1; j++) {
			for (int k=1; k<U->Nz; k++) {
				sum += 4.0*index3dR(U, c, j, k)*index3dR(U, c, j, k);
			}
		}
		// top and bottom boundaries
		for (int k=1; k<U->Nz; k++) {
			sum += 2.0*index3dR(U, c, 0, k)*index3dR(U, c, 0, k);
			sum += 2.0*index3dR(U, c, U->Ny-1, k)*index3dR(U, c, U->Ny-1, k);
		}

		// left and right boundaries (equal)
		for (int j=1; j<U->Ny-1; j++) {
			sum += 4.0*index3dR(U, c, j, 0)*index3dR(U, c, j, 0);
		}

		// corners
		sum += 2*index3dR(U, c, 0, 0)*index3dR(U, c, 0, 0);
		sum += 2*index3dR(U, c, U->Ny-1, 0)*index3dR(U, c, U->Ny-1, 0);
	}

	// step size
	double hy = 2.0/(U->Ny-1.0);
	double hz = params->L/(U->Nz);

	return 0.5*sum/4.0*hy*hz;
}

double CFL(struct RealField *U, struct Parameters *params) {

	// z step size
	double hz = params->L/(U->Nz);

	double CFL_max = 0.0;
	double CFL_current;


	// check on innner grid points only
	for (int j=1; j<U->Ny-1; j++) { 
		for (int k=0; k<U->Nz; k++) { 

			// v component
			// check sign of velocity
			if ( index(U, 1, j) > 0 ) {
				CFL_current = index3dC(U, 1, j, k)*params->dt/params->h[j];
			} else {
				CFL_current = index3dC(U, 1, j, k)*params->dt/params->h[j-1];
			}

			if (CFL_current > CFL_max) {
				CFL_max = CFL_current;
			}

			// w component
			CFL_current = index3dC(U, 2, j, k)*params->dt/hz;
			if (CFL_current > CFL_max) {
				CFL_max = CFL_current;
			}
		}
	}
	return CFL_max;
}

void enforceSymmetry(struct ComplexField *UK) {
	// Enforce symmetries on the solution
	// This is done on the intarnal grid points
	// only, because at the boundary we apply our
	// own boundary conditions.

	for (int j=1; j<UK->Ny-1; j++){
		for (int k=0; k<UK->Nz/2+1; k++){
			// on u kill the imaginary part
			index3dC(UK, 0, j, k) = creal(index3dC(UK, 0, j, k));

			// on psi kill the real part
			index3dC(UK, 2, j, k) = I*cimag(index3dC(UK, 2, j, k));
		}
	}
}