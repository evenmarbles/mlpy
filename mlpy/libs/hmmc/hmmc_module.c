#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"

#include "hmmc_module.h"
#include "hmm.h"

/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef HmmcMethods[] = {
	{ "forward", forward_wrapper, METH_VARARGS, "Perform forward algorithm" },
	{ "backward", backward_wrapper, METH_VARARGS, "Perform backward algorithm" },
	{ "fwdbkw", fwdbkw_wrapper, METH_VARARGS, "Perform forward/backward algorithm" },
	{ "computeTwoSliceSum", compueTwoSliceSum_wrapper, METH_VARARGS, "Compute the sum of the two-slice distibutions over hidden states" },
	{ "viterbi", viterbi_wrapper, METH_VARARGS, "Find the most likely path" },
	{ NULL, NULL, 0, NULL }
};

/* ==== Initialize the HMM functions ====================== */
DL_EXPORT(void) inithmmc(void)
{
	Py_InitModule("hmmc", HmmcMethods);
	import_array();
}

/* #### HMM Extensions ############################## */

static PyObject * forward_wrapper(PyObject * self, PyObject * args)
{
	int K = 0;
	int T = 0;
	int dims[2];

	double * init_state_distrib = NULL;
	double * transmat = NULL;
	double * obslik = NULL;
	double * alpha = NULL;

	double loglik = 0.0f;

	PyArrayObject *pyinit_state_distrib, *pytransmat, *pyobslik, *pyalpha;

	if (!PyArg_ParseTuple(args, "OOO", &pyinit_state_distrib, &pytransmat, &pyobslik)) {
		return NULL;
	}

	/* Check that object input is 'double' type and a vector
	Not needed if python wrapper function checks before call to this routine */
	if (not_doublevector(pyinit_state_distrib)) return NULL;

	/* Check that object input is double type and a 2D array
	Not needed if python wrapper function checks before call to this routine */
	if (not_double2Darray(pytransmat)) return NULL;
	if (not_double2Darray(pyobslik)) return NULL;

	/* Get the dimensions of the input */
	K = PyArray_DIM(pyinit_state_distrib, 0);
	if (PyArray_DIM(pytransmat, 0) != K)
		PyErr_SetString(PyExc_ValueError, "The transition matrix must be of size KxK.");
	if (PyArray_DIM(pytransmat, 1) != K)
		PyErr_SetString(PyExc_ValueError, "The transition matrix must be of size KxK.");

	dims[0] = PyArray_DIM(pyobslik, 0);
	if (dims[0] != K)
		PyErr_SetString(PyExc_ValueError, "The obslik must have K rows.");
	T = dims[1] = PyArray_DIM(pyobslik, 1);

	/* Make a new double array of same dimension */
	pyalpha = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);

	/* Change contiguous arrays into C *arrays   */
	init_state_distrib = pyvector_to_Carrayptrs(pyinit_state_distrib);
	transmat = pyvector_to_Carrayptrs(pytransmat);
	obslik = pyvector_to_Carrayptrs(pyobslik);
	alpha = pyvector_to_Carrayptrs(pyalpha);

	loglik = forward(K, T, init_state_distrib, transmat, obslik, alpha);

	Py_DECREF(init_state_distrib);
	Py_DECREF(transmat);
	Py_DECREF(obslik);
	Py_DECREF(alpha);

	return Py_BuildValue("fO", loglik, PyArray_Return(pyalpha));
}

static PyObject * backward_wrapper(PyObject * self, PyObject * args)
{
	int K = 0;
	int T = 0;
	int dims[2];

	double * transmat = NULL;
	double * obslik = NULL;
	double * alpha = NULL;
	double * gamma = NULL;
	double * beta = NULL;

	PyArrayObject *pytransmat, *pyobslik, *pygamma, *pyalpha, *pybeta;

	if (!PyArg_ParseTuple(args, "OOO", &pytransmat, &pyobslik, &pyalpha)) {
		return NULL;
	}

	/* Check that object input is double type and a 2D array
	Not needed if python wrapper function checks before call to this routine */
	if (not_double2Darray(pytransmat)) return NULL;
	if (not_double2Darray(pyobslik)) return NULL;
	if (not_double2Darray(pyalpha)) return NULL;

	/* Get the dimensions of the input */
	K = PyArray_DIM(pyalpha, 0);
	if (PyArray_DIM(pytransmat, 0) != K)
		PyErr_SetString(PyExc_ValueError, "The transition matrix must be of size KxK.");
	if (PyArray_DIM(pytransmat, 1) != K)
		PyErr_SetString(PyExc_ValueError, "The transition matrix must be of size KxK.");

	dims[0] = PyArray_DIM(pyobslik, 0);
	if (dims[0] != K)
		PyErr_SetString(PyExc_ValueError, "The obslik must have K rows.");
	T = dims[1] = PyArray_DIM(pyobslik, 1);

	if (PyArray_DIM(pyalpha, 0) != K)
		PyErr_SetString(PyExc_ValueError, "The alpha matrix must have K rows.");
	if (PyArray_DIM(pyalpha, 1) != T)
		PyErr_SetString(PyExc_ValueError, "The alpha matrix must have T colunns.");

	/* Make a new double vector of same dimension */
	pygamma = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);
	pybeta = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);

	/* Change contiguous arrays into C *arrays   */
	transmat = pyvector_to_Carrayptrs(pytransmat);
	obslik = pyvector_to_Carrayptrs(pyobslik);
	alpha = pyvector_to_Carrayptrs(pyalpha);
	gamma = pyvector_to_Carrayptrs(pygamma);
	beta = pyvector_to_Carrayptrs(pybeta);

	backward(K, T, transmat, obslik, alpha, gamma, beta);

	Py_DECREF(transmat);
	Py_DECREF(obslik);
	Py_DECREF(alpha);
	Py_DECREF(gamma);
	Py_DECREF(beta);

	return Py_BuildValue("OO", PyArray_Return(pybeta), PyArray_Return(pygamma));
}

static PyObject * fwdbkw_wrapper(PyObject * self, PyObject * args)
{
	int K = 0;
	int T = 0;
	int tmp, dims[2];

	double * init_state_distrib = NULL;
	double * transmat = NULL;
	double * obslik = NULL;
	double * alpha = NULL;
	double * gamma = NULL;
	double * beta = NULL;

	double loglik = 0.0f;

	PyArrayObject *pyinit_state_distrib, *pytransmat, *pyobslik, *pygamma, *pyalpha, *pybeta;

	if (!PyArg_ParseTuple(args, "OOO", &pyinit_state_distrib, &pytransmat, &pyobslik)) {
		return NULL;
	}

	/* Check that object input is 'double' type and a vector
	Not needed if python wrapper function checks before call to this routine */
	if (not_doublevector(pyinit_state_distrib)) return NULL;

	/* Check that object input is double type and a 2D array
	Not needed if python wrapper function checks before call to this routine */
	if (not_double2Darray(pytransmat)) return NULL;
	if (not_double2Darray(pyobslik)) return NULL;

	/* Get the dimensions of the input */
	K = PyArray_DIM(pyinit_state_distrib, 0);
	if (PyArray_DIM(pytransmat, 0) != K)
		PyErr_SetString(PyExc_ValueError, "The transition matrix must be of size KxK.");
	if (PyArray_DIM(pytransmat, 1) != K)
		PyErr_SetString(PyExc_ValueError, "The transition matrix must be of size KxK.");

	tmp = dims[0] = PyArray_DIM(pyobslik, 0);
	if (tmp != K)
		PyErr_SetString(PyExc_ValueError, "The obslik must have K rows.");
	T = dims[1] = PyArray_DIM(pyobslik, 1);

	/* Make a new double vector of same dimension */
	pygamma = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);
	pyalpha = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);
	pybeta = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);

	/* Change contiguous arrays into C *arrays   */
	init_state_distrib = pyvector_to_Carrayptrs(pyinit_state_distrib);
	transmat = pyvector_to_Carrayptrs(pytransmat);
	obslik = pyvector_to_Carrayptrs(pyobslik);
	alpha = pyvector_to_Carrayptrs(pyalpha);
	gamma = pyvector_to_Carrayptrs(pygamma);
	beta = pyvector_to_Carrayptrs(pybeta);

	loglik = fwdbkw(K, T, init_state_distrib, transmat, obslik, gamma, alpha, beta);

	Py_DECREF(init_state_distrib);
	Py_DECREF(transmat);
	Py_DECREF(obslik);
	Py_DECREF(alpha);
	Py_DECREF(gamma);
	Py_DECREF(beta);

	return Py_BuildValue("OOOf", PyArray_Return(pygamma), PyArray_Return(pyalpha), PyArray_Return(pybeta), loglik);
}

static PyObject * compueTwoSliceSum_wrapper(PyObject * self, PyObject * args)
{
	int dims[2];
	int K = 0;
	int T = 0;

	int t, i, j;                       /* loop indices */
	int ndx;
	double xitSum;

	double * alpha = NULL;
	double * beta = NULL;
	double * transmat = NULL;
	double * obslik = NULL;
	double * xisummed = NULL;

	double * b = NULL;
	double * xit = NULL;	/* temporary storage */

	PyArrayObject *pyalpha, *pybeta, *pytransmat, *pyobslik, *pyxisummed;

	if (!PyArg_ParseTuple(args, "OOOO", &pyalpha, &pybeta, &pytransmat, &pyobslik)) {
		return NULL;
	}

	/* Check that object input is double type and a 2D array
	Not needed if python wrapper function checks before call to this routine */
	if (not_double2Darray(pyalpha)) return NULL;
	if (not_double2Darray(pybeta)) return NULL;
	if (not_double2Darray(pytransmat)) return NULL;
	if (not_double2Darray(pyobslik)) return NULL;

	/* Get the dimensions of the input */
	dims[2];
	K = dims[0] = dims[1] = PyArray_DIM(pyalpha, 0);
	T = PyArray_DIM(pyalpha, 1);
	if (PyArray_DIM(pybeta, 0) != K || PyArray_DIM(pybeta, 1) != T)
		PyErr_SetString(PyExc_ValueError, "Input sizes must agree.");
	if (PyArray_DIM(pytransmat, 0) != K || PyArray_DIM(pytransmat, 1) != K)
		PyErr_SetString(PyExc_ValueError, "Input sizes must agree.");
	if (PyArray_DIM(pyobslik, 0) != K || PyArray_DIM(pyobslik, 1) != T)
		PyErr_SetString(PyExc_ValueError, "Input sizes must agree.");

	/* Make a new double vector of same dimension */
	pyxisummed = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);

	/* Change contiguous arrays into C *arrays   */
	alpha = pyvector_to_Carrayptrs(pyalpha);
	beta = pyvector_to_Carrayptrs(pybeta);
	transmat = pyvector_to_Carrayptrs(pytransmat);
	obslik = pyvector_to_Carrayptrs(pyobslik);
	xisummed = pyvector_to_Carrayptrs(pyxisummed);

	b = (double *)malloc(K*sizeof(double));
	xit = (double *)malloc(K*K*sizeof(double));	/* temporary storage */

	for (t = T - 2; t >= 0; --t)  {
		for (j = 0; j < K; ++j) {
			ndx = j*T + (t + 1);
			b[j] = beta[ndx] * obslik[ndx];
		}
		xitSum = 0;
		for (i = 0; i < K; ++i) {
			for (j = 0; j < K; ++j) {
				ndx = i*K + j;
				xit[ndx] = transmat[ndx] * alpha[i*T + t] * b[j];
				xitSum += xit[ndx];
			}
		}
		for (i = 0; i<K; ++i) {
			for (j = 0; j < K; ++j) {
				ndx = i*K + j;
				xisummed[ndx] += xit[ndx] / xitSum;
			}
		}
	}

	free(b);
	free(xit);

	Py_DECREF(alpha);
	Py_DECREF(beta);
	Py_DECREF(transmat);
	Py_DECREF(obslik);
	Py_DECREF(xisummed);

	return PyArray_Return(pyxisummed);
}

static PyObject * viterbi_wrapper(PyObject * self, PyObject * args)
{
	return Py_BuildValue("f", 0.0f);
}



/* #### Vector Utility functions ######################### */

/* ==== Make a Python Array Obj. from a PyObject, ================
generates a double vector w/ contiguous memory which may be a new allocation if
the original was not a double type or contiguous
!! Must DECREF the object returned from this routine unless it is returned to the
caller of this routines caller using return PyArray_Return(obj) or
PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pyvector(PyObject *objin)  {
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, 1, 1);
}
/* ==== Create 1D Carray from PyArray ======================
Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
	//int i, n;

	//n = PyArray_DIM(arrayin, 0);
	return (double *)PyArray_DATA(arrayin);  /* pointer to arrayin data as double */
}
/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
return 1 if an error and raise exception */
int  not_doublevector(PyArrayObject *vec)  {
	if (PyArray_TYPE(vec) != NPY_DOUBLE || PyArray_NDIM(vec) != 1)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublevector: array must be of type Float and 1 dimensional (n).");
		return 1;
	}
	return 0;
}

/* #### Matrix Utility functions ######################### */

/* ==== Make a Python Array Obj. from a PyObject, ================
generates a double matrix w/ contiguous memory which may be a new allocation if
the original was not a double type or contiguous
!! Must DECREF the object returned from this routine unless it is returned to the
caller of this routines caller using return PyArray_Return(obj) or
PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pydouble2Darray(PyObject *objin)  {
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, 2, 2);
}
/* ==== Create Carray from PyArray ======================
Assumes PyArray is contiguous in memory.
Memory is allocated!                                    */
double **pydouble2Darray_to_Carrayptrs(PyArrayObject *arrayin)  {
	double **c, *a;
	int i, n, m;

	n = PyArray_DIM(arrayin, 0);
	m = PyArray_DIM(arrayin, 1);
	c = ptrvector(n);
	a = (double *)PyArray_DATA(arrayin);  /* pointer to arrayin data as double */
	for (i = 0; i<n; i++)  {
		c[i] = a + i*m;
	}
	return c;
}
/* ==== Allocate a double *vector (vec of pointers) ======================
Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
	double **v;
	v = (double **)malloc((size_t)(n*sizeof(double)));
	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);
	}
	return v;
}
/* ==== Free a double *vector (vec of pointers) ========================== */
void free_Carrayptrs(double **v)  {
	free((char*)v);
}
/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
return 1 if an error and raise exception */
int  not_double2Darray(PyArrayObject *mat)  {
	if (PyArray_TYPE(mat) != NPY_DOUBLE || PyArray_NDIM(mat) != 2)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_double2Darray: array must be of type Float and 2 dimensional (n x m).");
		return 1;
	}
	return 0;
}
