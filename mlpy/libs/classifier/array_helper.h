#ifndef ARRAY_HELPER_H
#define ARRAY_HELPER_H

#include "Python.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_Classifier
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"

	/* .... C vector utility functions ..................*/
PyArrayObject *pyvector(PyObject *objin, int N, int M);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin, int N, int M);
int  not_doublevector(PyArrayObject *vec);

/* .... C 2D int array utility functions ..................*/
PyArrayObject *pydouble2Darray(PyObject *objin, int N, int M);
double **pydouble2Darray_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrdoublevector(long n);
void free_Cdouble2Darrayptrs(double **v);
int  not_double2Darray(PyArrayObject *mat);

#endif
