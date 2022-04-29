#ifndef __field_h__
#define __field_h__

#include <complex.h>
#include <fftw3.h>


struct ComplexField {
	/*  This data structure contains three arrays for the fourier modes
		of the field, and its size.  The arrays are stored in memory as 
		three blocks of memory, of fftw_complex numbers. Each element is composed
		of two double precision numbers. Data in these arrays is stored
		in row major or C order. We use the indexing macro index3dC defined below
		to access specific elements in matrix notation, like for example

		index3dC(U_k, 0, j, k) is mode associated to wavenumber vector (j, k)
		of component u

		ComplexField instanced are used for everything we have in Fourier space,
		such as fourier modes of velocity, wavenumber vector array, and other secondary stuff.
	*/
	int Ny, Nz;
	fftw_complex *component[3]; // Fourier coefficients for the three components
};

struct RealField {
	/*  This data structure contains three arrays for the grid point values
		of the field, and its size.  The arrays are stored in memory as 
		three blocks of memory, of double precision numbers. Data in these 
		arrays is stored in row major or C order. We use the indexing 
		macro index3dR defined below to access specific elements in matrix notation.

		index3dR(U, 0, j, k) is the value of the component u at grid point j, k
	*/
	int Ny, Nz;
	double *component[3]; // data in physical space for the three components
};

#include "parameters.h"
#include "fftwplans.h"


/*******************/
/* Indexing macros */
/*******************/
// Get element j, k of component i of the complex field cf.
// data is stored in row-major order

// for real fields
#define index3dR(cf, i, j, k) (cf->component[i][(j)*(cf->Nz) + (k)]) 

// for complex fields
#define index3dC(cf, i, j, k) (cf->component[i][(j)*(cf->Nz/2+1) + (k)])

// Get element j of buffer i. This macro indexes along the buffer as a 
// one dimensional array
#define index(cf, ii, jj) (cf->component[ii][jj])


/***************************/
/*  ComplexField Functions */
/***************************/
struct ComplexField *complexFieldCreate(int Ny, int Nz);
void complexFieldDestroy(struct ComplexField *field);
void complexFieldCopy(struct ComplexField *dest, struct ComplexField *src);
void complexFieldPrint(struct ComplexField *field, int slice);


/************************/
/*  RealField Functions */
/************************/
struct RealField *realFieldCreate(int Ny, int Nz);
void realFieldDestroy(struct RealField *field);

/********************/
/*  Other functions */
/********************/
struct RealField *uniformGrid2DCreate(struct Parameters *p);

/**********************/
/*  Padding functions */
/**********************/
void padComplexField(struct ComplexField *toBePadded, struct ComplexField *padded, int My, int Mz);
void cropComplexField(struct ComplexField *toBeCropped, struct ComplexField *out);
void truncateComplexField(struct ComplexField *U);

#endif