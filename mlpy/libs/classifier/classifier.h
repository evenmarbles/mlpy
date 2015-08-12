#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "Python.h"
#include "structmember.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_Classifier
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"


/*
* PyClassPair_Type
*/
typedef struct _classpairobject PyClassPairObject;
struct _classpairobject {
	PyObject_HEAD
	PyArrayObject *in_; /* tree input */
	double out;	/* tree output */
};

extern PyTypeObject PyClassPair_Type;

#define PyClassPair_Check(op) PyObject_TypeCheck(op, &PyClassPair_Type)
#define PyClassPair_CheckExact(op) (Py_TYPE(op) == &PyClassPair_Type)



/*
* PyClassPairList_Type
*/
typedef struct _classpairlistobject PyClassPairListObject;
struct _classpairlistobject {
	PyListObject list;
};

extern PyTypeObject PyClassPairList_Type;

#define PyClassPairList_Check(op) PyObject_TypeCheck(op, &PyClassPairList_Type)
#define PyClassPairList_CheckExact(op) (Py_TYPE(op) == &PyClassPairList_Type)

#endif	// CLASSIFIER_H
