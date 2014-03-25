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
void initSolution(struct ComplexField *UK, struct Parameters *params, struct FFTWPlans *plans) {
	/*	Initialize complex field from a binary file. 
	*/

	char initfile[1000];

	// start from scratch
	if (params->t_restart == 0.0) {
		sprintf(initfile, "init");
	} else {
		sprintf(initfile, "%f/data", params->t_restart);
	}

	// check size of file first, to be sure we have the exact sizes
	int size = fsize(initfile);
	// it can be twice of five time the nominal size
	if (size != 2*UK->Ny*UK->Nz*sizeof(double)) {
		if (size != 5*UK->Ny*UK->Nz*sizeof(double)) {
			log_err("The size of the init file does not appear to match the number"
				"of modes specified. Aborting gracefully.");
			exit(1);
		}
	}

	// open file
	FILE *fh = fopen(initfile, "r");

	// create storage since data file is in physical space
    struct RealField *U  = realFieldCreate(params->Ny, params->Nz);

	// read only azimuthal velocity and streamfunction, which are the two first
	// we assume the rest is bullshit since we only need the two variables.
    fread(U->component[0], sizeof(double), U->Ny*U->Nz, fh);
    fread(U->component[2], sizeof(double), U->Ny*U->Nz, fh);

    // go to fourier
    fft(U, UK, plans);

    // at this point we enforce the boundary conditions for the three
    // components of the solution, u, omega, and psi. 
    enforceNoSlip(UK, params);

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

	fclose(fh);
}
