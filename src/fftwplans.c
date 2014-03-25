#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include <string.h>
#include <omp.h>

#include "operators.h"
#include "fftwplans.h"
#include "field.h"

struct FFTWPlans *fftwPlansCreate(int Nz) {
	/*	Creates a struct that wraps pointers to data
		and fft plans alltogheter, so that it is easy to carry 
		it around. The data structure is defined in the header file
		fftwplans.h

		Notes
		-----
		This struct is to be used with the fft and ifft function
		in this same file.

	*/

	// create data structure
	struct FFTWPlans *plans = malloc(sizeof(struct FFTWPlans));

	// inverse fft - fourier to physical
	plans->ifft_in  = fftw_alloc_complex(Nz/2+1);
	plans->ifft_out = fftw_alloc_real(Nz);
	plans->ifft     = fftw_plan_dft_c2r_1d(Nz,
                                           plans->ifft_in,
                                           plans->ifft_out, 
                                           FFTW_MEASURE);

	// forward fft - physical to fourier
	plans->fft_in  = fftw_alloc_real(Nz);
	plans->fft_out = fftw_alloc_complex(Nz/2+1);
	plans->fft     = fftw_plan_dft_r2c_1d(Nz,
			   	     			          plans->fft_in, 
                                          plans->fft_out,
                                          FFTW_MEASURE);
    // size of the transform
    plans->Nz = Nz;

	return plans;
}

void fftwPlansDestroy(struct FFTWPlans *plans){
	/*	Clean up memory when everything is over.

		Parameters
		----------
		struct FFTWPlans *plans: pointer to the struct containing plans and arrays

	*/
	fftw_free(plans->ifft_out);
	fftw_free(plans->ifft_in);
	fftw_free(plans->fft_out);
	fftw_free(plans->fft_in);
	fftw_destroy_plan(plans->fft);
	fftw_destroy_plan(plans->ifft);
	free(plans);
}

void fft(struct RealField *U, struct ComplexField *UK, struct FFTWPlans *plans){
	/*	Transform data from physical space to Fourier space. */

	for (int i=0; i<3; i++) {
		for (int j=0; j<UK->Ny; j++) {
			memcpy(plans->fft_in, &index3dR(U, i, j, 0), sizeof(double)*U->Nz);
			fftw_execute(plans->fft);
			memcpy(&index3dC(UK, i, j, 0), plans->fft_out, sizeof(fftw_complex)*U->Nz/2+1);
		}
	}
}

void ifft(struct ComplexField *UK, struct RealField *U, struct FFTWPlans *plans){
	/*	Transform data from Fourier space to physical space. */
	for (int i=0; i<3; i++) {
		for (int j=0; j<UK->Ny; j++) {
			memcpy(plans->ifft_in, &index3dC(UK, i, j, 0), sizeof(fftw_complex)*UK->Nz/2+1);
			fftw_execute(plans->ifft);
			memcpy(&index3dR(U, i, j, 0), plans->ifft_out, sizeof(double)*U->Nz);
		}
	}
	// Now normalize transforms
	double den = 1.0/U->Nz;
	for(int i=0; i<3; i++){
		for(int j=0; j<U->Ny*U->Nz; j++){
			index(U, i, j) *= den;
		}
	}
}