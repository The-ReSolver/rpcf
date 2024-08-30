#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <signal.h>
#include <omp.h>
#include <unistd.h>

#include "dbg.h"
#include "parameters.h"
#include "operators.h"
#include "control.h"
#include "field.h"
#include "fftwplans.h"
#include "output.h"

#ifndef MAX_BUF
#define MAX_BUF 200
#endif



int main(int argc, char **argv) {

	// return code of the program
	int return_code = 0;

	// parse command line arguments
	if (argc == 2){
		chdir(argv[1]);
	}

	/********************************/
	/*        Checks     		    */
	/********************************/
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

	// buffer to keep the recent time history of the kinetic energy
	struct Buffer *Ks  = bufferCreate(params->dt);

	// these two will contain the velocity field in fourier and physical
    struct RealField *V          = realFieldCreate(params->Ny, params->Nz);
    struct ComplexField *VK      = complexFieldCreate(params->Ny, params->Nz);
    // and this is the transform of the solution in physical space
    struct RealField *U          = realFieldCreate(params->Ny, params->Nz);

	/********************/	
	/* Initialization   */
	/********************/
	double w0 = 0;
	double psi_upper = 0;

	initSolution(UK, params, plans, w0, psi_upper);
	complexFieldCopy(UK_old, UK);

	// Here i have a flow solution that satisfies the boundary conditions, and where 
	// vorticity has been obtained by solution of the poisson equation.

	// compute nonlinear term first
	nonLinearTerm(UK, storage_r0, storage_r1, storage_r2, params, plans, NK);

	/******************/
	/* Output headers */
	/******************/	
	printf("Time        K                  dKdt                dKdt/K              CFL\n");
	printf("----------------------------------------------------------------------------------\n");

	/*************************/
	/* Main integration loop */
	/*************************/
	double t = params->t_restart;
	for (int it=0; it<(params->T - params->t_restart)/params->dt; it++) {

		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
		// ~~~~~~~~~    CONTROL METHOD    ~~~~~~~~~~~~~~~~ //
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
		//w0 = params->A*sin(params->eta*t);

		// chirp signal
		// w0 = params->A*sin(0.001*pow(1.022, t));

		// constant bottom wall velocity
		w0 = 0.0;

		// then apply bc
		// applyBC(UK, params, w0, psi_upper);

		// general bc not applied as control aspects of code ignored
		enforceNoSlip(UK, params);

		// update buffer for derivative of the kinetic energy
		if ((it + 2) % params->n_it_out == 0 ||
			(it + 2) % params->n_it_print == 0 ||
			(it + 1) % params->n_it_out == 0 ||
			(it + 1) % params->n_it_print == 0 ||
			(it + 0) % params->n_it_out == 0 ||
			(it + 0) % params->n_it_print == 0) {

			// get velocity components
			toVelocity(UK, VK, storage_c0, params, w0);

			// get velocity components in physical space
			ifft(VK, V, plans);

			updateBuffer(Ks, integralKineticEnergy(V, params));

			// save data to disk
			if ((it + 0) % params->n_it_out == 0 && (t >= params->t_offset || t == 0.0)) {
				ifft(UK, U, plans);

				// save data to file
				if (params->output_mode == 1){
					saveSnapshot(t, U, V);
					saveMetadata(t, Ks->data[0], ddt(Ks));
				}
				else if (params->output_mode == 2){
					saveKineticEnergy(t, Ks->data[0]);
				}

				if (params->steady_halt && fabs(ddt(Ks)) < 1e-12 && t != 0.0) {
					break;
				}
			}

			// print data to stdout
			if ((it + 0) % params->n_it_print == 0) {
				printf("%.5e %.12e %+.12e %+.12e %.5e\n", t, Ks->data[0],
														ddt(Ks),
														ddt(Ks)/Ks->data[0],
														CFL(V, params));
				fflush(stdout);
			}
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
		solveVorticityStreamFuncHelmoltzProblems(RK, params, UK, w0, psi_upper);

		// update current time 
		t += params->dt;

		// enlarge the domain
		// params->Re = 10 + 0.2*t;
		//params->L = 8.0 + t*(80-8)/10000.0;
		//params->alpha = 4*asin(1.0)/params->L;

		// reload params file in case it's been updated
		reloadParametersFromParamsFile(params);
	}


	/***********************/
	/* Save final snapshot */
	/***********************/
	if (params->output_mode == 2){
		saveSnapshot(t, U, V);
		saveMetadata(t, Ks->data[0], ddt(Ks));
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
