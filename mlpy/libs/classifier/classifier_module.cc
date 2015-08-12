#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Python.h"
#include "structmember.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_Classifier
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"

#include "random.h"
#include "classifier.h"
#include "c45tree.h"

#ifdef __cplusplus
extern "C" {
#endif

	/* ==== Set up the methods table ====================== */
	static PyMethodDef ClassifierMethods[] = {
		{ NULL, NULL, 0, NULL }
	};

	/* ==== Initialize the C45 functions ====================== */
#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
	PyMODINIT_FUNC initclassifier(void)
	{
		PyObject *m;

		m = Py_InitModule3("classifier", ClassifierMethods, "Classification module");
		if (m == NULL) return;

		if (PyType_Ready(&PyRandom_Type) < 0)
			return;
		Py_INCREF(&PyRandom_Type);
		PyModule_AddObject(m, "Random", (PyObject *)&PyRandom_Type);

		if (PyType_Ready(&PyClassPair_Type) < 0)
			return;
		Py_INCREF(&PyClassPair_Type);
		PyModule_AddObject(m, "ClassPair", (PyObject *)&PyClassPair_Type);

		PyClassPairList_Type.tp_base = &PyList_Type;
		if (PyType_Ready(&PyClassPairList_Type) < 0)
			return;
		Py_INCREF(&PyClassPairList_Type);
		PyModule_AddObject(m, "ClassPairList", (PyObject *)&PyClassPairList_Type);

		if (PyType_Ready(&PyTreeNode_Type) < 0)
			return;
		Py_INCREF(&PyTreeNode_Type);
		PyModule_AddObject(m, "C45TreeNode", (PyObject *)&PyTreeNode_Type);

		if (PyType_Ready(&PyTreeExperience_Type) < 0)
			return;
		Py_INCREF(&PyTreeExperience_Type);
		PyModule_AddObject(m, "TreeExperience", (PyObject *)&PyTreeExperience_Type);

		if (PyType_Ready(&PyC45Tree_Type) < 0)
			return;
		Py_INCREF(&PyC45Tree_Type);
		PyModule_AddObject(m, "C45Tree", (PyObject *)&PyC45Tree_Type);

		import_array();
	}

#ifdef __cplusplus
}
#endif
