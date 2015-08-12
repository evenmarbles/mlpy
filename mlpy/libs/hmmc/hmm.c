#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Python.h"
#include "hmm.h"

#undef max


double forward(int K, int T, double * init_state_distrib, double * transmat, double * obslik, double * alpha)
{
	int t = 0;

	double scale = 0.0f;
	double loglik = 0.0f;
	
	double * transmatT = (double *)malloc(K*K*sizeof(double));
	double *m = (double *)malloc(K*sizeof(double)); /* temporary quantities in the algorithm */

	/* the tranposed version of transmat*/
	transposeSquareInPlace(transmatT, transmat, K, K);

	multiplyInPlace(alpha, init_state_distrib, obslik, K, T, 1, T);
	scale = normalizeInPlace(alpha, K, T);
	loglik = log(scale);

	for (t = 1; t<T; ++t){
		multiplyMatrixInPlace(m, transmatT, alpha + (t - 1), K, T);
		multiplyInPlace(alpha + t, m, obslik + t, K, T, 1, T);
		scale = normalizeInPlace(alpha + t, K, T);
		loglik += log(scale);
	}

	free(m);
	free(transmatT);

	return loglik;
}

void backward(int K, int T, double * transmat, double * obslik, double * alpha, double * gamma, double * beta)
{
	int t, d;

	/* special care for eta since it has 3 dimensions */
	int eta_ndim = 3;
	int * eta_dims = (int *)malloc(eta_ndim*sizeof(int));

	/* temporary quantities in the algorithm */
	double *m = (double *)malloc(K*sizeof(double));
	double *b = (double *)malloc(K*sizeof(double));			// (K-1)*(T+1)
	double *eta = (double *)malloc(K*K*T*sizeof(double));
	double *squareSpace = (double *)malloc(K*K*sizeof(double));

	t = T - 1;
	/* I don't think we need to initialize beta to all zeros. */
	for (d = 0; d<K; ++d) {
		beta[d*T + t] = 1;
		gamma[d*T + t] = alpha[d*T + t];
	}

	///* Put the last slice of eta as zeros, to be compatible with Sohrab and Gavin's code.
	//There are no values to put there anyways. This means that you can't normalise the
	//last matrix in eta, but it shouldn't be used. Note the d<K*K range.
	//*/
	//for (d = 0; d<(K*K); ++d) {
	//	/*mexPrintf("setting *(eta + %d) = 0 \n", d+t*K*K);*/
	//	*(eta + d*T + t) = 0;/*(double)7.0f;*/
	//}

	/* We have to remember that the 1:T range in Matlab is 0:(T-1) in C. */
	for (t = (T - 2); t >= 0; --t) {

		/* setting beta */
		multiplyInPlace(b, beta + t + 1, obslik + t + 1, K, 1, T, T);
		/* Using "m" again instead of defining a new temporary variable.
		   We using a lot of lines to say
		   beta(:,t) = normalize(transmat * b);
		*/
		multiplyMatrixInPlace(m, transmat, b, K, 1);
		normalizeInPlace(m, K, 1);
		for (d = 0; d<K; ++d) { beta[d*T + t] = m[d]; }
		/* using "m" again as valueholder */

		///* setting eta, whether we want it or not in the output */
		//outerProductUVInPlace(squareSpace, alpha + t, b, K, T);
		//componentVectorMultiplyInPlace(eta + t, transmat, squareSpace, K, T);
		//normalizeInPlace(eta + t, K*K, T);

		/* setting gamma */
		multiplyInPlace(m, alpha + t, beta + t, K, 1, T, T);
		normalizeInPlace(m, K, 1);
		for (d = 0; d<K; ++d) { gamma[d*T + t] = m[d]; }
	}

	free(b); free(m); free(squareSpace);
	free(eta); free(eta_dims);

	return;
}

double fwdbkw(int K, int T, double * init_state_distrib, double * transmat, double * obslik,
	double * gamma, double * alpha, double * beta)
{
	/********* Forward. ********/
	double loglik = forward(K, T, init_state_distrib, transmat, obslik, alpha);

	/********* Backward. ********/
	backward(K, T, transmat, obslik, alpha, gamma, beta);

	return loglik;
}


/* .... C array manipulation utility functions ..................*/

/*
	Normalize column in place and return the normalization constant used.
*/
double normalizeInPlace(double * A, unsigned int nrows, unsigned int stride)
{
	unsigned int n;
	double sum = 0;

	for (n = 0; n<nrows; ++n)
	{
		sum += A[n*stride];
		if (A[n*stride] < 0)
		{
			PyErr_SetString(PyExc_ValueError,
				"We don't want to normalize if A contains a negative value. This is a logical error.");
		}
	}

	if (sum > 0)
	{
		for (n = 0; n<nrows; ++n)
			A[n*stride] /= sum;
	}
	return sum;
}

void multiplyInPlace(double * result, double * u, double *v, unsigned int nrows, unsigned int rstride, unsigned int ustride, unsigned int vstride)
{
	unsigned int n;

	for (n = 0; n < nrows; ++n)
	{
		result[n*rstride] = u[n*ustride] * v[n*vstride];
	}
}

void multiplyMatrixInPlace(double * result, double * A, double * v, unsigned int nrows, unsigned int stride)
{
	unsigned int i, d;

	for (d = 0; d<nrows; ++d) {
		result[d] = 0;
		for (i = 0; i<nrows; ++i){
			result[d] += A[d*nrows + i] * v[i*stride];
		}
	}
}

void transposeSquareInPlace(double * out, double * in, unsigned int K, unsigned int T)
{
	unsigned int i, j, n;

	for (n = 0; n<K*T; n++) {
		i = n / K;
		j = n%K;
		out[n] = in[T*j + i];
	}
}

void outerProductUVInPlace(double * out, double * u, double * v, unsigned int K, unsigned int T)
{
	unsigned int i, j;

	for (i = 0; i<K; ++i){
		for (j = 0; j<K; ++j){
			out[i*K + j] = u[i*T] * v[j*T];
		}
	}
	return;
}

void componentVectorMultiplyInPlace(double * out, double * u, double * v, unsigned int K, unsigned int T)
{
	unsigned int i, j;

	for (i = 0; i < K; ++i) {
		for (j = 0; j < K; ++j){
			out[i*T + j*K*T] = u[i*K + j] * v[i*K + j];
		}
	}

	return;
}
