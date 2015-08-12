#include "array_helper.h"


/* #### Vector Utility functions ######################### */

/* ==== Make a Python Array Obj. from a PyObject, ================
generates a double vector w/ contiguous memory which may be a new allocation if
the original was not a double type or contiguous
!! Must DECREF the object returned from this routine unless it is returned to the
caller of this routines caller using return PyArray_Return(obj) or
PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pyvector(PyObject *objin, int N, int M)
{
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, N, M);
}

/* ==== Create 1D Carray from PyArray ======================
Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin, int N, int M)
{
	PyArrayObject *pyarrayin = arrayin;
	if (N != 1 && M != 1) {
		pyarrayin = pyvector((PyObject *)arrayin, N, M);
	}
	return (double *)PyArray_DATA(pyarrayin);  /* pointer to arrayin data as double */
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
return 1 if an error and raise exception */
int  not_doublevector(PyArrayObject *vec)
{
	if (PyArray_TYPE(vec) != NPY_DOUBLE || PyArray_NDIM(vec) != 1)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublevector: array must be of type Float and 1 dimensional (n).");
		return 1;
	}
	return 0;
}

/* #### Double Array Utility functions ######################### */

/* ==== Make a Python double Array Obj. from a PyObject, ================
generates a 2D integer array w/ contiguous memory which may be a new allocation if
the original was not an integer type or contiguous
!! Must DECREF the object returned from this routine unless it is returned to the
caller of this routines caller using return PyArray_Return(obj) or
PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pydouble2Darray(PyObject *objin, int N, int M)
{
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, N, M);
}

/* ==== Create double 2D Carray from PyArray ======================
Assumes PyArray is contiguous in memory.
Memory is allocated!                                    */
double **pydouble2Darray_to_Carrayptrs(PyArrayObject *arrayin)
{
	double **c, *a;
	int i, n, m;

	n = PyArray_DIMS(arrayin)[0];
	m = PyArray_DIMS(arrayin)[1];
	c = ptrdoublevector(n);
	a = (double *)PyArray_DATA(arrayin);  /* pointer to arrayin data as double */
	for (i = 0; i<n; i++)  {
		c[i] = a + i*m;
	}
	return c;
}

/* ==== Allocate a a *int (vec of pointers) ======================
Memory is Allocated!  See void free_Carray(int ** )                  */
double **ptrdoublevector(long n)
{
	double **v;
	v = (double **)malloc((size_t)(n*sizeof(double)));
	if (!v)   {
		printf("In **ptrintvector. Allocation of memory for int array failed.");
		exit(0);
	}
	return v;
}

/* ==== Free an int *vector (vec of pointers) ========================== */
void free_Cdouble2Darrayptrs(double **v)
{
	free((char*)v);
}

/* ==== Check that PyArrayObject is an int (integer) type and a 2D array ==============
return 1 if an error and raise exception
Note:  Use NPY_DOUBLE for NumPy integer array, not NP_INT      */
int not_double2Darray(PyArrayObject *mat)
{
	if (PyArray_TYPE(mat) != NPY_DOUBLE || PyArray_NDIM(mat) != 2)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_double2Darray: array must be of type int and 2 dimensional (n x m).");
		return 1;
	}
	return 0;
}
