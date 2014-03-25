#ifndef __parameters_h__
#define __parameters_h__

// This struct contains all the parameters which we want to use in the 
// simulation so we carry around a single variable, instead of many more.
struct Parameters {
	int Ny, Nz;    // number of modes along the two directions
	double Re;     // Reynolds number
	double Ro;     // rotation number
	double dt;     // time step
	double T;      // total integration time
	double alpha;  // axial wavenumber
	double L;      // axial wavenumber
	int n_it_out;  // save a file each n_it_out iterations
	float t_restart; // if different from zero, we will try to restart the simulation
				     // from the snapshot at this time. 
	float stretch_factor; // factor for hyperbolic tangent stretching
	double *h;  // spacings of the grid
	int n_threads;  // number of openmp threads
};

struct Parameters *loadParametersFromParamsFile(void);
struct Parameters *loadParametersFromCommandLine(int argc, char *argv[]);
void parametersDestroy(struct Parameters *params);

#endif