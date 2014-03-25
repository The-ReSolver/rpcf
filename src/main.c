#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <signal.h>
#include <omp.h>

#include "dbg.h"
#include "parameters.h"
#include "operators.h"
#include "control.h"
#include "field.h"
#include "fftwplans.h"
#include "output.h"



int main(int argc, char **argv) {

	// return code of the program
	int return_code = 0;


	/********************************/
	/*        Checks     		    */
	/********************************/
	if (!fileExists("init")){
		log_info("init file does not exist in the current folder. Exiting gracefully.");
		return_code = 1;
		exit(return_code);
	}

	if (!fileExists("params")){
		log_info("params file does not exist in the current folder. Exiting gracefully.");
		return_code = 1;
		exit(return_code);
	}
	
	/********************************/
	/* Parse command line arguments */
	/********************************/
	struct Parameters *params = loadParametersFromParamsFile();
	omp_set_num_threads(params->n_threads);

	
	/*********************/
	/* Create fftw plans */
	/*********************/
	struct FFTWPlans *plans = fftwPlansCreate(params->Nz);

	
	/**********************/
	/* Physical variables */
	/**********************/
	// velocity/vorticity/streamfunction
    struct ComplexField *UK      = complexFieldCreate(params->Ny, params->Nz);
    struct ComplexField *UK_old  = complexFieldCreate(params->Ny, params->Nz);
    
    // nonlinear term
	struct ComplexField *NK      = complexFieldCreate(params->Ny, params->Nz);
	struct ComplexField *NK_old  = complexFieldCreate(params->Ny, params->Nz);

	// right hand side of helmoltz problems
	struct ComplexField *RK      = complexFieldCreate(params->Ny, params->Nz);

	// temporary block of memory
	struct RealField    *storage_r0 = realFieldCreate(params->Ny, params->Nz);
	struct RealField    *storage_r1 = realFieldCreate(params->Ny, params->Nz);
	struct RealField    *storage_r2 = realFieldCreate(params->Ny, params->Nz);
	struct ComplexField *storage_c0 = complexFieldCreate(params->Ny, params->Nz);

	// other structures
	struct Buffer *Ks  = bufferCreate(params->dt);

	// these two will contain the velocity field in fourier and physical
    struct RealField *V          = realFieldCreate(params->Ny, params->Nz);
    struct ComplexField *VK      = complexFieldCreate(params->Ny, params->Nz);
    // and this is the transform of the solution in physical space
    struct RealField *U          = realFieldCreate(params->Ny, params->Nz);

	/********************/	
	/* Initialization   */
	/********************/
	initSolution(UK, params, plans);
	complexFieldCopy(UK_old, UK);

	// Here i have a flow solution that satisfies no-slip boundary conditions, and where 
	// vorticity has been obtained by solution of the poisson equation.

	// compute nonlinear term first
	nonLinearTerm(UK, storage_r0, storage_r1, storage_r2, params, plans, NK);

	/*************************/
	/* Main integration loop */
	/*************************/
	double t = params->t_restart;
	for (int it=0; it<(params->T - params->t_restart)/params->dt; it++) {

		// update buffer for derivative
		if ((it + 2) % params->n_it_out == 0 || 
			(it + 1) % params->n_it_out == 0 ||
			(it + 0) % params->n_it_out == 0) {

			// get velocity components
			toVelocity(UK, VK, storage_c0, params);

			// get velocity components in physical space
			ifft(VK, V, plans);

			updateBuffer(Ks, integralKineticEnergy(V, params));

			if ( (it + 0) % params->n_it_out == 0) {
				ifft(UK, U, plans);

				// save data to file
				saveSnapshot(t, U, V);
				saveMetadata(t, Ks->data[0], ddt(Ks));

				// output to screen
				printf("%.5e %.12e %+.12e %+.12e %.5e\n", t, Ks->data[0], 
										       	 	         ddt(Ks), 
											     	         ddt(Ks)/Ks->data[0],///log(10),
											     	         CFL(V, params));
				fflush(stdout);
			}
		}

		// then apply bc
		if (t<0) {
			// wall_normal_openloop(UK, params, 1, 0.05*params->Nz);
			wall_normal_opposition(UK, params, 0);
		} else {
			enforceNoSlip(UK, params);
		}

		// update old nonlinear term
		complexFieldCopy(NK_old, NK);

		// compute nonlinear term
		nonLinearTerm(UK, storage_r0, storage_r1, storage_r2, params, plans, NK);

		// compute rhs 
		computeRHSHelmoltz(UK, UK_old, NK, NK_old, params, storage_c0, RK);

		// store solution at previous step, before updating it.
		complexFieldCopy(UK_old, UK);

		// solve helmoltz problems
		solveVelocityHelmoltzProblems(RK, params, UK);
		solveVorticityStreamFuncHelmoltzProblems(RK, params, UK);

		// update current time 
		t += params->dt;

		// increase Re 
		// params->Re = 1000.0 + t/20;

		// exit condition
		// if ( fabs(ddt(Ks)) < 1e-12 ) {
		// break;
		// }
	}



	/********************/	
	/* Free stuff       */
	/********************/
	complexFieldDestroy(UK);
	complexFieldDestroy(UK_old);
	complexFieldDestroy(NK);
	complexFieldDestroy(NK_old);
	complexFieldDestroy(RK);
	complexFieldDestroy(VK);
	complexFieldDestroy(storage_c0);

	realFieldDestroy(storage_r0);
	realFieldDestroy(storage_r1);
	realFieldDestroy(storage_r2);
	realFieldDestroy(U);
	realFieldDestroy(V);

	parametersDestroy(params);
	fftwPlansDestroy(plans);
	free(Ks);
	return return_code;	
}
