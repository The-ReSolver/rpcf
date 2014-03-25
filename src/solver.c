#include <complex.h>

void solve_tridiagonal(complex *lower, complex *maind, complex *upper, complex *x, int N) {
	/*	Solver tridiagonal linear system using gaussian elimination
	*/

	// forward pass
	for (int i=1; i<N; i++) {
		maind[i] = maind[i]*maind[i-1]/lower[i-1] - upper[i-1];
		upper[i] = upper[i]*maind[i-1]/lower[i-1];
		x[i] = x[i]*maind[i-1]/lower[i-1] - x[i-1];
	}

	// backward pass
	x[N-1] = x[N-1]/maind[N-1];
	for (int i=N-2; i>=0; i--){
		x[i] = x[i]/maind[i] - upper[i]/maind[i]*x[i+1];
	}
}