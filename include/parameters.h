#ifndef __parameters_h__
#define __parameters_h__

// This struct contains all the parameters which we want to use in the 
// simulation so we carry around a single variable, instead of many more.
struct Parameters {
	int Ny, Nz;     // number of modes along the two directions
	double Re;      // Reynolds number
	double Ro;      // rotation number
	double dt;      // time step
	double T;       // total integration time
	double alpha;   // axial wavenumber
	double L;       // axial wavenumber
	int n_it_out;   // save a file each n_it_out iterations
	int n_it_print; // print flow information to stdout every n_it_print iterations
					// defaults to n_it_out if not specified in params file
	double t_restart; // if different from zero, we will try to restart the simulation
				     // from the snapshot at this time. 
	double t_offset; // amount of time before we start to write data
	double stretch_factor; // factor for hyperbolic tangent stretching
	int output_mode; // type of data the solver outputs at each iteration
					 // output_mode = 1: full data
					 // output_mode = 2: kinetic energy only
	// int bctype;    // type of boundary conditions.
	//			   // if 0 we assume a zero net mass flux, so psi on the walls is 0
	//			   // if 1 we assume a zero pressure gradient, so psi on the top wall
	//			   // will come out of the simulation.	
	// double A;      // amplitude of the oscillatory motion
	// double eta;    // angular frequency of the oscillatory motion
	double *h;  // spacings of the grid
	int n_threads;  // number of openmp threads
	int steady_halt; // halt the run if a steady solution is found
};

struct Parameters *loadParametersFromParamsFile(void);
void reloadParametersFromParamsFile(struct Parameters *params);
struct Parameters *loadParametersFromCommandLine(int argc, char *argv[]);
void parametersDestroy(struct Parameters *params);

#endif