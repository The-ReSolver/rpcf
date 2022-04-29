#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>

#include "parameters.h"
#include "dbg.h"
#include "field.h"
#include "operators.h"
#include "fftwplans.h"
#include "output.h"


/**************************/
/* ComplexField routines */
/**************************/
struct ComplexField *complexFieldCreate(int Ny, int Nz) {
	/*	Allocate memory for a ComplexField instance and memset values to zero.

		Parameters
		----------
		int Ny, Nz : number of modes along the y and z directions

		Returns
		-------
		*out : pointer to Field instance or NULL pointer in case 
			   of allocation errors
	*/

	// allocate space variables
	struct ComplexField *cf = malloc(sizeof(struct ComplexField));
	check(cf, "Could not allocate memory for field. Aborting.");

	// set sizes
	cf->Ny = Ny;       
	cf->Nz = Nz;       

	// allocate the rest	
	cf->component[0] = malloc(sizeof(fftw_complex)*cf->Ny*(cf->Nz/2+1));
	check(cf->component[0], "Could not allocate memory for modes of the first velocity component.");
	memset(cf->component[0], 0, cf->Ny*(cf->Nz/2+1)*sizeof(fftw_complex));

	cf->component[1] = malloc(sizeof(fftw_complex)*cf->Ny*(cf->Nz/2+1));
	check(cf->component[1], "Could not allocate memory for modes of the second velocity component.");
	memset(cf->component[1], 0, cf->Ny*(cf->Nz/2+1)*sizeof(fftw_complex));

	cf->component[2] = malloc(sizeof(fftw_complex)*cf->Ny*(cf->Nz/2+1));
	check(cf->component[2], "Could not allocate memory for modes of the third velocity component.");
	memset(cf->component[2], 0, cf->Ny*(cf->Nz/2+1)*sizeof(fftw_complex));

	return cf;

	error:
		return NULL;
}

void complexFieldDestroy(struct ComplexField *cf) {
	/*	Deallocate memory.

		Parameters
		----------
		struct ComplexField *cf : pointer to ComplexField structure
	*/
	for(int i = 0; i < 3; i++){
		fftw_free(cf->component[i]); 
	}
	free(cf);
}

void complexFieldCopy(struct ComplexField *dest, struct ComplexField *src) {
	/* Create a duplicate of a ComplexField.
	*/

	dest->Ny = src->Ny;       
	dest->Nz = src->Nz;       
	memcpy(dest->component[0], src->component[0], sizeof(fftw_complex)*dest->Ny*(dest->Nz/2+1));
	memcpy(dest->component[1], src->component[1], sizeof(fftw_complex)*dest->Ny*(dest->Nz/2+1));
	memcpy(dest->component[2], src->component[2], sizeof(fftw_complex)*dest->Ny*(dest->Nz/2+1));
}

void complexFieldPrint(struct ComplexField *cf, int component) {
	/*	Print a component of ComplexField to the screen.
		
		Parameters
		----------
		struct ComplexField *cf : pointer to ComplexField structure
		int component : the index of the component to be printed

	*/
	for(int j=0; j<cf->Ny; j++){
		for(int k=0; k<cf->Nz/2+1; k++){
            printf("%9.4f ", creal(index3dC(cf, component, j, k)));
		}
		printf("\n");
	}
}

/**************************/
/* Real Field routines */
/**************************/
struct RealField *realFieldCreate(int Ny, int Nz) {
	/*	Allocate memory for RealField instance and memset memory to zero.

		Parameters
		----------
		int Ny, Nz : number of modes along the y and z directions

		Returns
		-------
		*out : pointer to Field instance or NULL pointer in case 
			   of allocation errors
	*/

	// declare variable
	struct RealField *rf = malloc(sizeof(struct RealField));
	check(rf, "Could not allocate memory for field. Aborting.");

	// set sizes
	rf->Ny = Ny;       
	rf->Nz = Nz;    
	
	rf->component[0] = fftw_alloc_real(rf->Ny*rf->Nz);
	check(rf->component[0], "Could not allocate memory for modes of the first field component.");
	memset(rf->component[0], 0, rf->Ny*rf->Nz*sizeof(double));

	rf->component[1] = fftw_alloc_real(rf->Ny*rf->Nz);
	check(rf->component[1], "Could not allocate memory for modes of the second field component.");
	memset(rf->component[1], 0, rf->Ny*rf->Nz*sizeof(double));

	rf->component[2] = fftw_alloc_real(rf->Ny*rf->Nz);
	check(rf->component[2], "Could not allocate memory for modes of the third field component.");
	memset(rf->component[2], 0, rf->Ny*rf->Nz*sizeof(double));

	return rf;

	error:
		return NULL;
}

void realFieldDestroy(struct RealField *rf) {
	/*	Deallocate memory.

		Parameters
		----------
		struct RealField *rf : pointer to RealField structure
	*/
	for (int i=0; i<3; i++) {
		fftw_free(rf->component[i]); 
	}
	free(rf);
}


/********************/
/* Padding routines */
/********************/
// these are used for de-aliased calculations
void padComplexField(struct ComplexField *toBePadded, struct ComplexField *padded, int My, int Mz) {
	/*	Pad data.

		Notes
		-----
		The output is the zero padded version of the input. Padding 
		is done in the "middle" of the array, at those positions
		corresponding to the higher wavenumbers. 
	*/

	//first quadrant, top-left
	for (int i=0; i<3; i++) {
		for (int j=0; j<toBePadded->Ny/2+1; j++) {
			for (int k=0; k<toBePadded->Nz/2+1; k++) {
				index3dC(padded, i, j, k) = index3dC(toBePadded, i, j, k);
			}
		}
	}

	//second quadrant, bottom-left
	for (int i=0; i<3; i++) {
		for (int j=toBePadded->Ny/2; j<toBePadded->Ny; j++) {
			for (int k=0; k<toBePadded->Nz/2+1; k++) {
				index3dC(padded, i, j + (My - toBePadded->Ny), k) = index3dC(toBePadded, i, j, k);
			}
		}
	}
}

void truncateComplexField(struct ComplexField *U) {
	/* */

	int M = U->Ny;
	int N = 2*M/3;

	// remove rows
	for (int i=0; i<3; i++) {
		for (int j=N/2; j<M-N/2; j++) {
			for (int k=0; k<M/2+1; k++) {
				index3dC(U, i, j, k) = 0.0 + 0.0*I;
			}
		}
	}

	// remove columns
	for (int i=0; i<3; i++) {
		for (int j=0; j<M; j++) {
			for (int k=N/2; k<M/2+1; k++) {
				index3dC(U, i, j, k) = 0.0 + 0.0*I;
			}
		}
	}
}

void cropComplexField(struct ComplexField *toBeCropped, struct ComplexField *cropped) {
	/*	The inverse operation of padding. Basically we select only
		Fourier modes with wavenumbers less than the original size,
		and we discard higher wavenumbers which were originated by 
		operating on padded data.
	*/
	//first quadrant, top-left
	for (int i=0; i<3; i++) {
		for (int j=0; j<cropped->Ny/2+1; j++) {
			for (int k=0; k<cropped->Nz/2+1; k++) {
				index3dC(cropped, i, j, k) = index3dC(toBeCropped, i, j, k);
			}
		}
	}

	//second quadrant, bottom-left
	for (int i=0; i<3; i++) {
		for (int j=cropped->Ny/2; j<cropped->Ny; j++) {
			for (int k=0; k<cropped->Nz/2+1; k++) {
				index3dC(cropped, i, j, k) = index3dC(toBeCropped, i, j+(toBeCropped->Ny - cropped->Ny), k);
			}
		}
	}
}
