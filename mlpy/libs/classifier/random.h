#ifndef RANDOM_H
#define RANDOM_H

#include "Python.h"
#include "structmember.h"

#include <fstream>
#include <iostream>
#include <vector>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include "math.h"
#include <process.h> /* for getpid() and the exec..() family */
#else
#include <unistd.h>   // for getpid
#endif

#include "coord.h"

using namespace std;


static const long   _M = 0x7fffffff; // 2147483647 (Mersenne prime 2^31-1)
static const long   A_ = 0x10ff5;    // 69621
static const long   Q_ = 0x787d;     // 30845
static const long   R_ = 0x5d5e;     // 23902

static const float _F = (float)(1. / _M);;
static const short _NTAB = 32;         // arbitrary length of shuffle table
static const long  _DIV = 1 + (_M - 1) / _NTAB;


/*
* PyRandom_Type 
*/

typedef struct PyRandom {
	PyObject_HEAD
	PyListObject* _table;            // shuffle table of seeds
	long         _next;              // seed to be used as index into table
	long         _seed;              // current random number seed
	unsigned     _seed2;             // seed for tausworthe random bit
#ifndef _WIN32
	pthread_mutex_t random_mutex;
#endif
} PyRandomObject;

extern PyTypeObject PyRandom_Type;

#define PyRandom_Check(op) PyObject_TypeCheck(op, &PyRandom_Type)
#define PyRandom_CheckExact(op) (Py_TYPE(op) == &PyRandom_Type)

// utility functions
void reset(PyRandomObject *self);

// Continuous Distributions
float arcsine(PyRandomObject *self, float xMin = 0., float xMax = 1.);
float beta(PyRandomObject *self, float v, float w, float xMin = 0., float xMax = 1.);
float cauchy(PyRandomObject *self, float a = 0., float b = 1.);
float chiSquare(PyRandomObject *self, int df);
float cosine(PyRandomObject *self, float xMin = 0., float xMax = 1.);
float floatLog(PyRandomObject *self, float xMin = -1., float xMax = 1.);
float erlang(PyRandomObject *self, float b, int c);
float exponential(PyRandomObject *self, float a = 0., float c = 1.);
float extremeValue(PyRandomObject *self, float a = 0., float c = 1.);
float fRatio(PyRandomObject *self, int v, int w);
float gamma(PyRandomObject *self, float a, float b, float c);
float laplace(PyRandomObject *self, float a = 0., float b = 1.);
float logarithmic(PyRandomObject *self, float xMin = 0., float xMax = 1.);
float logistic(PyRandomObject *self, float a = 0., float c = 1.);
float lognormal(PyRandomObject *self, float a, float mu, float sigma);
float normal(PyRandomObject *self, float mu = 0., float sigma = 1.);
float parabolic(PyRandomObject *self, float xMin = 0., float xMax = 1.);
float pareto(PyRandomObject *self, float c);
float pearson5(PyRandomObject *self, float b, float c);
float pearson6(PyRandomObject *self, float b, float v, float w);
float power(PyRandomObject *self, float c);
float rayleigh(PyRandomObject *self, float a, float b);
float studentT(PyRandomObject *self, int df);
float triangular(PyRandomObject *self, float xMin = 0., float xMax = 1., float c = 0.5);
float uniform(PyRandomObject *self, float xMin = 0., float xMax = 1.);
float userSpecified(
	PyRandomObject *self,
	float(*usf)(                // pointer to user-specified function
	float,						// x
	float,						// xMin
	float),						// xMax
	float xMin, float xMax,     // function domain
	float yMin, float yMax);
float weibull(PyRandomObject *self, float a, float b, float c);

// Discrete Distributions
bool bernoulli(PyRandomObject *self, float p = 0.5);
int binomial(PyRandomObject *self, int n, float p);
int geometric(PyRandomObject *self, float p);
int hypergeometric(PyRandomObject *self, int n, int N, int K);
void multinomial(PyRandomObject *self, int n, float p[], int count[], int m);
int negativeBinomial(PyRandomObject *self, int s, float p);
int pascal(PyRandomObject *self, int s, float p);
int poisson(PyRandomObject *self, float mu);
int uniformDiscrete(PyRandomObject *self, int i, int j);

// Sampling
float sample(PyRandomObject *self, bool replace = true);
void sample(PyRandomObject *self, float x[], int ndim);

// Stochastic Interpolation
cartesianCoord stochasticInterpolation(void);

// Multivariate Distributions
cartesianCoord bivariateNormal(PyRandomObject *self,
	float muX = 0.,
	float sigmaX = 1.,
	float muY = 0.,
	float sigmaY = 1.);
cartesianCoord bivariateUniform(PyRandomObject *self,
	float xMin = -1.,
	float xMax = 1.,
	float yMin = -1.,
	float yMax = 1.);
cartesianCoord corrNormal(PyRandomObject *self,
	float r,
	float muX = 0.,
	float sigmaX = 1.,
	float muY = 0.,
	float sigmaY = 1.);
cartesianCoord corrUniform(PyRandomObject *self,
	float r,
	float xMin = 0.,
	float xMax = 1.,
	float yMin = 0.,
	float yMax = 1.);
sphericalCoord spherical(PyRandomObject *self,
	float thMin = 0.,
	float thMax = M_PI,
	float phMin = 0.,
	float phMax = 2. * M_PI);
void sphericalND(PyRandomObject *self, float x[], int n);

// Number Theoretic Distributions
float avoidance(void);
void avoidance(float x[], unsigned ndim);
bool tausworthe(PyRandomObject *self, unsigned n);
void tausworthe(PyRandomObject *self, bool *bitvec, unsigned n);

#endif		// RANDOM_H
