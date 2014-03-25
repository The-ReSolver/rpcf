#ifndef __fftwplans_h__
#define __fftwplans_h__ 

#include <complex.h>
#include <fftw3.h>


// This is a structure which contains information for
// performing ffts, pointers to arrays and plans
// which is carried around the code to compute fast fourier
// transforms.
struct FFTWPlans{

    int Nz;

	// fourier to physical plan and pointer to arrays 
	fftw_plan ifft;
	fftw_complex *ifft_in;
	double *ifft_out;

	// physical to fourier plan and poiters to arrays
	fftw_plan fft;
	double *fft_in;
	fftw_complex *fft_out;
};


#include "parameters.h"
#include "field.h"

struct FFTWPlans *fftwPlansCreate(int Nz);
void fftwPlansDestroy(struct FFTWPlans *plans);
void fft(struct RealField *U, struct ComplexField *U_K, struct FFTWPlans *plans);
void ifft(struct ComplexField *U_K, struct RealField *U, struct FFTWPlans *plans);

void ifft_worker_function(complex *in, double *out, int Nz);


#endif