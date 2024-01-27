#include <stdio.h>
#include <complex.h>
#include <fftw3.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "field.h"
#include "dbg.h"
#include "output.h"
#include "control.h"
#include "operators.h"

#define true 1
#define false 0

/*********************/
/* Buffer functions  */
/*********************/
struct Buffer *bufferCreate(double dt) {
	/*	Allocate memory for object.
	*/
	struct Buffer *buf = malloc(sizeof(struct Buffer));
	buf->data[0] = 0.0;
	buf->data[1] = 0.0;
	buf->data[2] = 0.0;
	buf->dt = dt;
	return buf;
}

void updateBuffer(struct Buffer *buf, double value) {
	/*	 A function to update a buffer struct
	*/
	buf->data[2] = buf->data[1];
	buf->data[1] = buf->data[0];
	buf->data[0] = value;
}

double ddt(struct Buffer *buf) {
	/*	Compute derivative of data in the buffer,
		at this time.
	*/
	return (3*buf->data[0] - 4*buf->data[1] + buf->data[2])/2/buf->dt;
}

/*********************/
/* Output functions  */
/*********************/

void saveSnapshot(double t, struct RealField *U, struct RealField *V) {
	/*	Save output to binary file.	*/

	char somebuffer[1000];

	// make snapshot directory
	sprintf(somebuffer, "%f", t);
	mkdir(somebuffer, 0700);

	/* SAVE VELOCITY COMPONENTS */
	sprintf(somebuffer, "%f/U", t);
	FILE *fh = fopen(somebuffer, "w");
	dumpToBinary(V->component[0], fh, U->Ny, U->Nz);
	dumpToBinary(V->component[1], fh, U->Ny, U->Nz);
	dumpToBinary(V->component[2], fh, U->Ny, U->Nz);
	fclose(fh);

	/* SAVE VORTICITY */
	sprintf(somebuffer, "%f/omega", t);
	fh = fopen(somebuffer, "w");
	dumpToBinary(U->component[1], fh, U->Ny, U->Nz);
	fclose(fh);

	/* SAVE STREAM FUNCTION */
	sprintf(somebuffer, "%f/psi", t);
	fh = fopen(somebuffer, "w");
	dumpToBinary(U->component[2], fh, U->Ny, U->Nz);
	fclose(fh);
}

void dumpToBinary(double *data, FILE *fh, int Ny, int Nz) {
	/* 	Dump data in a binary file, writing also data for 
		the right boundary, where we have periodic boundary 
		conditions
	*/
	for (int i=0; i<Ny*Nz; i=i+Nz) {
		fwrite(&data[i], sizeof(double), Nz, fh);
		fwrite(&data[i], sizeof(double), 1, fh);
	}
}

void saveMetadata(double t, double K, double dKdt) {
	/* Write metadata to metadata file */
	// build filename from data directory and time
	char somebuffer[1000];
	sprintf(somebuffer, "%f/metadata", t);

	// open file
	FILE *fh = fopen(somebuffer, "w");
	fprintf(fh, "[metadata]\n");
	fprintf(fh, "t = %.16e\n", t);
	fprintf(fh, "K = %.16e\n", K);
	fprintf(fh, "dKdt = %.16e\n", dKdt);

	fclose(fh);
}

void saveKineticEnergy(double t, double K){
	/*  Write kinetic energy and associated time to separate
	    files
	*/
	char somebuffer[1000];

	// append kinetic energy to file
	sprintf(somebuffer, "K");
	FILE *fh = fopen(somebuffer, "a");
	fwrite(&K, sizeof(double), 1, fh);
	fclose(fh);

	// append current time to file
	sprintf(somebuffer, "t");
	fh = fopen(somebuffer, "a");
	fwrite(&t, sizeof(double), 1, fh);
	fclose(fh);
}

int fileExists(const char *filename) {
	/*	Check if a file exists
	*/
	FILE *fileh = fopen(filename, "r");
    if (fileh) {
        fclose(fileh);
        return true;
    }
    return false;
}

int fsize(const char *filename) {
	/*	Return size in bytes of a file.
	 */
	struct stat st;

	if (stat(filename, &st) == 0) {
		return st.st_size;
	}

	return -1;
}

/*******************/
/* Input functions */
/*******************/
void initSolution(struct ComplexField *UK, struct Parameters *params, struct FFTWPlans *plans, double w0, double psi_upper) {
	//	Initialize solution from snapshot directory. 
	//  We assume at this stage that the wall are steady, 
	// 	and control is zero.

	// strings to store the filename of the data files
	// we'll use to initialise the solution.
	char U_file[1000];
	char psi_file[1000];
	sprintf(U_file, "%f/U", params->t_restart);
	sprintf(psi_file, "%f/psi", params->t_restart);

	// create storage since data file is in physical space
    struct RealField *U  = realFieldCreate(params->Ny, params->Nz);
	
	// check size of file first, to be sure we have the exact sizes
	// it must be three times the nominal size.
	// the nominal size contains a plus 1, since we also write 
	// the right boundary, to have the same data.
	int U_file_size = fsize(U_file);
	int psi_file_size = fsize(psi_file);
	if (    (U_file_size != 3*(UK->Nz + 1)*UK->Ny*sizeof(double)) ||
			(psi_file_size != (UK->Nz + 1)*UK->Ny*sizeof(double)) ) {
		log_err("The size of the data files does not appear to match the number "
				"of modes specified. Aborting gracefully.");
		exit(1);
	}

	// open files
	FILE *U_fh = fopen(U_file, "r");
	FILE *psi_fh = fopen(psi_file, "r");

	// since data is written with both the left and right 
	// boundaries we will skip some data every line.
	for (int j=0; j<params->Ny; j++) {
		fread(&(U->component[0][j*U->Nz]), sizeof(double), U->Nz, U_fh);
		fread(&(U->component[2][j*U->Nz]), sizeof(double), U->Nz, psi_fh);
		fseek(U_fh, sizeof(double), SEEK_CUR);
		fseek(psi_fh, sizeof(double), SEEK_CUR);
    }
    
    // go to fourier
    fft(U, UK, plans);

    // at this point we enforce the boundary conditions for the three
    // components of the solution, u, omega, and psi. WHY THIS?
    // because the input file might not satisfy the boundary conditions.
    // solution.
    applyBC(UK, params, w0, psi_upper);

	// Then we update omega, cause we didnt have it in the file.
	// For this we need the second derivative of the streamfunction
	struct ComplexField *storage = complexFieldCreate(UK->Ny, UK->Nz);
	computeD2DY2(UK, storage, params);

	// note we only update at the inner points, since we already have set the boundary condition
	for (int j=1; j<params->Ny-1; j++) {
		for (int k=0; k<params->Nz/2+1; k++) {
			index3dC(UK, 1, j, k) =   k*k*params->alpha*params->alpha*index3dC(UK, 2, j, k) 
									- index3dC(storage, 2, j, k);
		}
	}

	complexFieldDestroy(storage);
	realFieldDestroy(U);

	fclose(U_fh);
	fclose(psi_fh);
}
