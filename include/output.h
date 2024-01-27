#ifndef __output_h__
#define __output_h__

struct Buffer {
	double data[3];
	double dt;
};


#include "field.h"

// functions to operate on buffer
struct Buffer *bufferCreate(double dt);
void updateBuffer(struct Buffer *buf, double value);
double ddt(struct Buffer *buf);

// output functions
void saveSnapshot(double t, struct RealField *U, struct RealField *V);
void dumpToBinary(double *data, FILE *fh, int Ny, int Nz);
void saveMetadata(double t, double K, double dKdt);
void saveKineticEnergy(double t, double K);

int fileExists(const char * filename);
int fsize(const char *filename);

// input functions
void initSolution(struct ComplexField *UK, struct Parameters *params, struct FFTWPlans *plans, double w0, double psi_upper);

#endif