#include "array_helper.h"
#include "classifier.h"


/*
* PyClassPair_Type
*/

void PyClassPair_dealloc(PyClassPairObject* self)
{
	Py_XDECREF(self->in_);
	self->ob_type->tp_free((PyObject*)self);
}

PyObject * PyClassPair_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyClassPairObject *self;

	self = (PyClassPairObject *)type->tp_alloc(type, 0);
	if (self != NULL) {
		self->in_ = (PyArrayObject *)PyList_New(0);
		if (self->in_ == NULL)
		{
			Py_DECREF(self);
			return NULL;
		}

		self->out = 0;
	}

	return (PyObject *)self;
}

int PyClassPair_init(PyClassPairObject *self, PyObject *args, PyObject *kwds)
{
	PyArrayObject *in_ = NULL, *tmp;

	static char *kwlist[] = { "in_", "out", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O!d", kwlist,
		&PyArray_Type,
		&in_,
		&self->out))
		return -1;

	if (in_) {
		/* Check that object input is 'double' type and a vector */
		if (not_doublevector(in_)) return NULL;

		tmp = (PyArrayObject *)self->in_;
		Py_INCREF(in_);
		self->in_ = (PyArrayObject *)in_;
		Py_DECREF(tmp);
	}

	return 0;
}


PyArrayObject * PyClassPair_get_in_(PyClassPairObject *self, void *closure)
{
	Py_INCREF(self->in_);
	return self->in_;
}

int PyClassPair_set_in_(PyClassPairObject *self, PyArrayObject *value, void *closure)
{
	if (value == NULL) {
		PyErr_SetString(PyExc_TypeError, "Cannot delete the in_ attribute");
		return -1;
	}

	if (!PyArray_Check(value) || not_doublevector(value)) {
		PyErr_SetString(PyExc_TypeError, "Value must be of type ndarray.");
		return NULL;
	}

	Py_DECREF(self->in_);
	Py_INCREF(value);
	self->in_ = value;

	return 0;
}

PyObject * PyClassPair_get_out(PyClassPairObject *self, void *closure)
{
	return Py_BuildValue("f", self->out);;
}

int PyClassPair_set_out(PyClassPairObject *self, PyObject *value, void *closure)
{
	if (value == NULL) {
		PyErr_SetString(PyExc_TypeError, "Cannot delete the out attribute");
		return -1;
	}

	if (!PyFloat_Check(value)) {
		PyErr_SetString(PyExc_TypeError,
			"The out attribute value must be a float");
		return -1;
	}

	self->out = PyFloat_AsDouble(value);

	return 0;
}

static PyGetSetDef PyClassPair_getseters[] = {
	{ "in_",
	(getter)PyClassPair_get_in_, (setter)PyClassPair_set_in_,
	"tree input",
	NULL },
	{ "out",
	(getter)PyClassPair_get_out, (setter)PyClassPair_set_out,
	"tree output",
	NULL },
	{ NULL }  /* Sentinel */
};


/* Pickle strategy:
__reduce__ by itself doesn't support getting kwargs in the unpickle
operation so we define a __setstate__ that replaces all the information
about the partial.  If we only replaced part of it someone would use
it as a hook to do strange things.
*/

PyObject *classpair_reduce(PyClassPairObject *cpo, PyObject *unused)
{
	return Py_BuildValue("O(Nd)", Py_TYPE(cpo), PyArray_Dumps((PyObject *)cpo->in_, -1), cpo->out);
}

PyObject *classpair_setstate(PyClassPairObject *cpo, PyObject *state)
{
	PyObject *in_;
	if (!PyArg_ParseTuple(state, "Od", &in_, &cpo->out))
		return NULL;
	Py_XDECREF(cpo->in_);
	cpo->in_ = (PyArrayObject *)in_;
	Py_INCREF(in_);
	Py_RETURN_NONE;
}

PyObject * PyClassPair_copy(PyObject *o)
{
	if (o == NULL || !PyClassPair_Check(o)) {
		PyErr_BadInternalCall();
		return NULL;
	}
	return PyObject_CallFunction((PyObject *)(Py_TYPE(o)), "Od",
		((PyClassPairObject *)o)->in_, ((PyClassPairObject *)o)->out, NULL);
}


PyDoc_STRVAR(copy__doc__,
	"D.copy() -> a shallow copy of D");

static PyMethodDef PyClassPair_methods[] = {
	{ "__reduce__", (PyCFunction)classpair_reduce, METH_NOARGS },
	{ "__setstate__", (PyCFunction)classpair_setstate, METH_O },
	{ "copy", (PyCFunction)PyClassPair_copy, METH_NOARGS, copy__doc__ },
	{ NULL, NULL },
};


PyDoc_STRVAR(class_pair__doc__,
	"Training instances for classification models.\n"
	"ClassPair() -> new empty class pair\n"
	"ClassPair(input, output) -> new class pair initialized from input and output data");

PyTypeObject PyClassPair_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                               /*ob_size*/
	"ClassPair",					 /*tp_name*/
	sizeof(PyClassPairObject),       /*tp_basicsize*/
	0,                               /*tp_itemsize*/
	(destructor)PyClassPair_dealloc, /*tp_dealloc*/
	0,                               /*tp_print*/
	0,                               /*tp_getattr*/
	0,                               /*tp_setattr*/
	0,                               /*tp_compare*/
	0,                               /*tp_repr*/
	0,                               /*tp_as_number*/
	0,                               /*tp_as_sequence*/
	0,                               /*tp_as_mapping*/
	0,                               /*tp_hash */
	0,                               /*tp_call*/
	0,                               /*tp_str*/
	0,                               /*tp_getattro*/
	0,                               /*tp_setattro*/
	0,                               /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
	class_pair__doc__,               /* tp_doc */
	0,		                         /* tp_traverse */
	0,		                         /* tp_clear */
	0,		                         /* tp_richcompare */
	0,		                         /* tp_weaklistoffset */
	0,		                         /* tp_iter */
	0,		                         /* tp_iternext */
	PyClassPair_methods,			 /* tp_methods */
	0,								 /* tp_members */
	PyClassPair_getseters,           /* tp_getset */
	0,                               /* tp_base */
	0,                               /* tp_dict */
	0,                               /* tp_descr_get */
	0,                               /* tp_descr_set */
	0,                               /* tp_dictoffset */
	(initproc)PyClassPair_init,      /* tp_init */
	0,                               /* tp_alloc */
	PyClassPair_new,                 /* tp_new */
};



/*
* PyClassPairList_Type
*/

int PyClassPairList_init(PyClassPairListObject *self, PyObject *args, PyObject *kwds)
{
	PyObject *seq = NULL, *item;
	static char *kwlist[] = { "sequence", 0 };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:list", kwlist, &seq))
		return -1;

	if (seq != NULL) {
		seq = PyObject_GetIter(seq);
		if (seq) {
			while (item = PyIter_Next(seq)) {
				if (!PyClassPair_Check(item)) {
					Py_DECREF(seq);
					Py_DECREF(item);
					PyErr_SetString(PyExc_TypeError, "all items must be of type ClassPair");
					return -1;
				}
				Py_DECREF(item);
			}
			/* clean up */
			Py_DECREF(seq);
		}
	}

	if (PyList_Type.tp_init((PyObject *)self, args, kwds) < 0)
		return -1;

	return 0;
}


PyObject * PyClassPairList_insert(PyClassPairListObject *self, PyObject *args)
{
	Py_ssize_t i;
	PyObject *v;
	if (!PyArg_ParseTuple(args, "nO:insert", &i, &v))
		return NULL;
	if (!PyClassPair_Check(v)) {
		PyErr_SetString(PyExc_TypeError,
			"item must be of type ClassPair");
		return NULL;
	}
	if (PyList_Insert((PyObject *)self, i, v) == 0)
		Py_RETURN_NONE;
	return NULL;
}

PyObject * PyClassPairList_append(PyClassPairListObject *self, PyObject *v)
{
	if (!PyClassPair_Check(v)) {
		PyErr_SetString(PyExc_TypeError,
			"item must be of type ClassPair");
		return NULL;
	}
	if (PyList_Append((PyObject *)self, v) == 0)
		Py_RETURN_NONE;
	return NULL;
}

PyDoc_STRVAR(append__doc__,
	"L.append(object) -- append object to end");
PyDoc_STRVAR(insert__doc__,
	"L.insert(index, object) -- insert object before index");

static PyMethodDef PyClassPairList_methods[] = {
	{ "append", (PyCFunction)PyClassPairList_append, METH_O, append__doc__ },
	{ "insert", (PyCFunction)PyClassPairList_insert, METH_VARARGS, insert__doc__ },
	{ NULL, NULL },
};


PyDoc_STRVAR(class_pair_list_doc,
	"Training instances for classification models.\n"
	"ClassPairList() -> new empty class pair\n"
	"ClassPairList(iterable) -> new class pair list initialized from iterable's items");

PyTypeObject PyClassPairList_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                               /*ob_size*/
	"ClassPairList",				 /*tp_name*/
	sizeof(PyClassPairListObject),   /*tp_basicsize*/
	0,                               /*tp_itemsize*/
	0,								 /*tp_dealloc*/
	0,                               /*tp_print*/
	0,                               /*tp_getattr*/
	0,                               /*tp_setattr*/
	0,                               /*tp_compare*/
	0,                               /*tp_repr*/
	0,                               /*tp_as_number*/
	0,                               /*tp_as_sequence*/
	0,                               /*tp_as_mapping*/
	0,                               /*tp_hash */
	0,                               /*tp_call*/
	0,                               /*tp_str*/
	0,                               /*tp_getattro*/
	0,                               /*tp_setattro*/
	0,                               /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
	class_pair_list_doc,             /* tp_doc */
	0,		                         /* tp_traverse */
	0,		                         /* tp_clear */
	0,		                         /* tp_richcompare */
	0,		                         /* tp_weaklistoffset */
	0,		                         /* tp_iter */
	0,		                         /* tp_iternext */
	PyClassPairList_methods,		 /* tp_methods */
	0,								 /* tp_members */
	0,								 /* tp_getset */
	0,                               /* tp_base */
	0,                               /* tp_dict */
	0,                               /* tp_descr_get */
	0,                               /* tp_descr_set */
	0,                               /* tp_dictoffset */
	(initproc)PyClassPairList_init,  /* tp_init */
	0,                               /* tp_alloc */
	0,								 /* tp_new */
};
