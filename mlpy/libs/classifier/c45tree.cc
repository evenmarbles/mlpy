#include <algorithm>    // std::remove_if

#include "c45tree.h"
#include "classifier.h"
#include "array_helper.h"


/*
* PyTreeNode_Type
*/

static PyMemberDef PyTreeNode_members[] = {
	{ "id", T_INT, offsetof(PyTreeNodeObject, id), READONLY,
	"Tree node id" },
	{ "dim", T_INT, offsetof(PyTreeNodeObject, dim), READONLY,
	"Dimension" },
	{ "val", T_FLOAT, offsetof(PyTreeNodeObject, val), READONLY,
	"Tree node value" },
	{ "type", T_BOOL, offsetof(PyTreeNodeObject, type), READONLY,
	"Tree node type" },
	{ "outputs", T_OBJECT, offsetof(PyTreeNodeObject, outputs), READONLY,
	"Set of all outputs seen at this leaf/node" },
	{ "nInstance", T_INT, offsetof(PyTreeNodeObject, nInstance), READONLY,
	"Instance number" },
	{ "l", T_OBJECT, offsetof(PyTreeNodeObject, l), READONLY,
	"Left child" },
	{ "r", T_OBJECT, offsetof(PyTreeNodeObject, r), READONLY,
	"Right child" },
	{ "leaf", T_BOOL, offsetof(PyTreeNodeObject, leaf), READONLY,
	"Whether this node is a leave or not" },
	{ NULL }  /* Sentinel */
};


void PyTreeNode_dealloc(PyTreeNodeObject* self)
{
	Py_XDECREF(self->outputs);
	self->ob_type->tp_free((PyObject*)self);
}

PyObject * PyTreeNode_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyTreeNodeObject *self;

	self = (PyTreeNodeObject *)type->tp_alloc(type, 0);
	if (self != NULL) {
		self->id = 0;

		self->dim = -1;
		self->val = -1;
		self->type = INVALID;

		self->outputs = (PyDictObject *)PyDict_New();
		if (self->outputs == NULL)
		{
			Py_DECREF(self);
			return NULL;
		}
		self->nInstance = 0;

		self->l = NULL;
		self->r = NULL;

		self->leaf = true;
	}

	return (PyObject *)self;
}

int PyTreeNode_init(PyTreeNodeObject *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = { "id", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist,
		&self->id))
		return -1;

	return 0;
}

PyTypeObject PyTreeNode_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                               /*ob_size*/
	"TreeNode",						 /*tp_name*/
	sizeof(PyTreeNodeObject),        /*tp_basicsize*/
	0,                               /*tp_itemsize*/
	(destructor)PyTreeNode_dealloc,  /*tp_dealloc*/
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
	"PyTreeNode objects",            /* tp_doc */
	0,		                         /* tp_traverse */
	0,		                         /* tp_clear */
	0,		                         /* tp_richcompare */
	0,		                         /* tp_weaklistoffset */
	0,		                         /* tp_iter */
	0,		                         /* tp_iternext */
	0,								 /* tp_methods */
	PyTreeNode_members,              /* tp_members */
	0,								 /* tp_getset */
	0,                               /* tp_base */
	0,                               /* tp_dict */
	0,                               /* tp_descr_get */
	0,                               /* tp_descr_set */
	0,                               /* tp_dictoffset */
	(initproc)PyTreeNode_init,       /* tp_init */
	0,                               /* tp_alloc */
	PyTreeNode_new,                  /* tp_new */
};


/*
* PyTreeExperience_Type
*/

void PyTreeExperience_dealloc(PyTreeExperienceObject* self)
{
	Py_XDECREF(self->in_);
	self->ob_type->tp_free((PyObject*)self);
}

PyObject * PyTreeExperience_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyTreeExperienceObject *self;

	self = (PyTreeExperienceObject *)type->tp_alloc(type, 0);
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

int PyTreeExperience_init(PyTreeExperienceObject *self, PyObject *args, PyObject *kwds)
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


PyArrayObject * PyTreeExperience_get_in_(PyTreeExperienceObject *self, void *closure)
{
	Py_INCREF(self->in_);
	return self->in_;
}

int PyTreeExperience_set_in_(PyTreeExperienceObject *self, PyArrayObject *value, void *closure)
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

PyObject * PyTreeExperience_get_out(PyTreeExperienceObject *self, void *closure)
{
	return Py_BuildValue("f", self->out);;
}

int PyTreeExperience_set_out(PyTreeExperienceObject *self, PyObject *value, void *closure)
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

static PyGetSetDef PyTreeExperience_getseters[] = {
	{ "in_",
	(getter)PyTreeExperience_get_in_, (setter)PyTreeExperience_set_in_,
	"tree input",
	NULL },
	{ "out",
	(getter)PyTreeExperience_get_out, (setter)PyTreeExperience_set_out,
	"tree output",
	NULL },
	{ NULL }  /* Sentinel */
};


PyObject * PyTreeExperience_copy(PyObject *o)
{
	if (o == NULL || !PyTreeExperience_Check(o)) {
		PyErr_BadInternalCall();
		return NULL;
	}
	return PyObject_CallFunction((PyObject *)(Py_TYPE(o)), "Od",
		((PyTreeExperienceObject *)o)->in_, ((PyTreeExperienceObject *)o)->out, NULL);
}


PyDoc_STRVAR(copy__doc__,
	"D.copy() -> a shallow copy of D");

static PyMethodDef PyTreeExperience_methods[] = {
	{ "copy", (PyCFunction)PyTreeExperience_copy, METH_NOARGS, copy__doc__ },
	{ NULL, NULL },
};


PyDoc_STRVAR(tree_experience__doc__,
	"Training instances for C4.5 decision tree model.\n"
	"TreeExperience() -> new empty tree experience\n"
	"TreeExperience(input, output) -> new tree experience initialized from input and output data");

PyTypeObject PyTreeExperience_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                               /*ob_size*/
	"TreeExperience",				 /*tp_name*/
	sizeof(PyTreeExperienceObject),  /*tp_basicsize*/
	0,                               /*tp_itemsize*/
	(destructor)PyTreeExperience_dealloc, /*tp_dealloc*/
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
	tree_experience__doc__,          /* tp_doc */
	0,		                         /* tp_traverse */
	0,		                         /* tp_clear */
	0,		                         /* tp_richcompare */
	0,		                         /* tp_weaklistoffset */
	0,		                         /* tp_iter */
	0,		                         /* tp_iternext */
	PyTreeExperience_methods,		 /* tp_methods */
	0,								 /* tp_members */
	PyTreeExperience_getseters,      /* tp_getset */
	0,                               /* tp_base */
	0,                               /* tp_dict */
	0,                               /* tp_descr_get */
	0,                               /* tp_descr_set */
	0,                               /* tp_dictoffset */
	(initproc)PyTreeExperience_init, /* tp_init */
	0,                               /* tp_alloc */
	PyTreeExperience_new,            /* tp_new */
};



/*
* PyTreeExperienceList_Type
*/

int PyTreeExperienceList_init(PyTreeExperienceListObject *self, PyObject *args, PyObject *kwds)
{
	PyObject *seq = NULL, *item;
	static char *kwlist[] = { "sequence", 0 };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:list", kwlist, &seq))
		return -1;

	seq = PyObject_GetIter(seq);
	if (seq) {
		while (item = PyIter_Next(seq)) {
			if (!PyTreeExperience_Check(item)) {
				Py_DECREF(seq);
				Py_DECREF(item);
				PyErr_SetString(PyExc_TypeError, "all items must be of type TreeExperience");
				return -1;
			}
			Py_DECREF(item);
		}
		/* clean up */
		Py_DECREF(seq);
	}

	if (PyList_Type.tp_init((PyObject *)self, args, kwds) < 0)
		return -1;

	return 0;
}


PyObject * PyTreeExperienceList_insert(PyTreeExperienceListObject *self, PyObject *args)
{
	Py_ssize_t i;
	PyObject *v;
	if (!PyArg_ParseTuple(args, "nO:insert", &i, &v))
		return NULL;
	if (!PyTreeExperience_Check(v)) {
		PyErr_SetString(PyExc_TypeError,
			"item must be of type TreeExperience");
		return NULL;
	}
	if (PyList_Insert((PyObject *)self, i, v) == 0)
		Py_RETURN_NONE;
	return NULL;
}

PyObject * PyTreeExperienceList_append(PyTreeExperienceListObject *self, PyObject *v)
{
	if (!PyTreeExperience_Check(v)) {
		PyErr_SetString(PyExc_TypeError,
			"item must be of type TreeExperience");
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

static PyMethodDef PyTreeExperienceList_methods[] = {
	{ "append", (PyCFunction)PyTreeExperienceList_append, METH_O, append__doc__ },
	{ "insert", (PyCFunction)PyTreeExperienceList_insert, METH_VARARGS, insert__doc__ },
	{ NULL, NULL },
};


PyDoc_STRVAR(tree_experience_list_doc,
	"Training instances for C4.5 decision tree model.\n"
	"TreeExperienceList() -> new empty tree experience\n"
	"TreeExperienceList(iterable) -> new tree experience list initialized from iterable's items");

PyTypeObject PyTreeExperienceList_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                               /*ob_size*/
	"TreeExperienceList",			 /*tp_name*/
	sizeof(PyTreeExperienceListObject),   /*tp_basicsize*/
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
	tree_experience_list_doc,        /* tp_doc */
	0,		                         /* tp_traverse */
	0,		                         /* tp_clear */
	0,		                         /* tp_richcompare */
	0,		                         /* tp_weaklistoffset */
	0,		                         /* tp_iter */
	0,		                         /* tp_iternext */
	PyTreeExperienceList_methods,	 /* tp_methods */
	0,								 /* tp_members */
	0,								 /* tp_getset */
	0,                               /* tp_base */
	0,                               /* tp_dict */
	0,                               /* tp_descr_get */
	0,                               /* tp_descr_set */
	0,                               /* tp_dictoffset */
	(initproc)PyTreeExperienceList_init,  /* tp_init */
	0,                               /* tp_alloc */
	0,								 /* tp_new */
};



/*
* PyC45Tree_Type
*/

static PyMemberDef PyC45Tree_members[] = {
	{ "id", T_INT, offsetof(PyC45TreeObject, id), READONLY,
	"Tree id" },
	{ "mode", T_INT, offsetof(PyC45TreeObject, mode), READONLY,
	"Tree mode" },
	{ "freq", T_INT, offsetof(PyC45TreeObject, freq), READONLY,
	"Tree frequency" },
	{ "m", T_INT, offsetof(PyC45TreeObject, m), READONLY,
	"Maximum number of visits for state to receive RMAX bonus" },
	{ "featPct", T_FLOAT, offsetof(PyC45TreeObject, featPct), READONLY,
	"Feature percentage" },
	{ "allow_only_splits", T_BOOL, offsetof(PyC45TreeObject, allow_only_splits), READONLY,
	"Allow only splits" },
	{ "rng", T_OBJECT, offsetof(PyC45TreeObject, rng), READONLY,
	"Random number generator" },
	{ "nOutput", T_INT, offsetof(PyC45TreeObject, nOutput), READONLY,
	"Number of ouputs" },
	{ "hadError", T_BOOL, offsetof(PyC45TreeObject, hadError), READONLY,
	"Whether an error occured" },
	{ "maxNodes", T_INT, offsetof(PyC45TreeObject, maxNodes), READONLY,
	"Maximum number of nodes" },
	{ "totalNodes", T_INT, offsetof(PyC45TreeObject, totalNodes), READONLY,
	"Total number of nodes" },
	{ "nExperiences", T_INT, offsetof(PyC45TreeObject, nExperiences), READONLY,
	"Number of experiences" },
	{ NULL }  /* Sentinel */
};


PyObject * PyC45Tree_get_DTDEBUG(PyC45TreeObject *self, void *closure)
{
	PyObject *res = (self->DTDEBUG) ? Py_True : Py_False;
	Py_INCREF(res);
	return res;
}

int PyC45Tree_set_DTDEBUG(PyC45TreeObject *self, PyObject *value, void *closure)
{
	if (value == NULL) {
		PyErr_SetString(PyExc_TypeError, "Cannot delete the DTDEBUG attribute");
		return -1;
	}

	if (!PyBool_Check(value)) {
		PyErr_SetString(PyExc_TypeError,
			"The out attribute value must be a boolean");
		return -1;
	}

	self->DTDEBUG = (value == Py_True) ? true : false;

	return 0;
}

PyObject * PyC45Tree_get_SPLITDEBUG(PyC45TreeObject *self, void *closure)
{
	PyObject *res = (self->SPLITDEBUG) ? Py_True : Py_False;
	Py_INCREF(res);
	return res;
}

int PyC45Tree_set_SPLITDEBUG(PyC45TreeObject *self, PyObject *value, void *closure)
{
	if (value == NULL) {
		PyErr_SetString(PyExc_TypeError, "Cannot delete the DTDEBUG attribute");
		return -1;
	}

	if (!PyBool_Check(value)) {
		PyErr_SetString(PyExc_TypeError,
			"The out attribute value must be a boolean");
		return -1;
	}

	self->SPLITDEBUG = (value == Py_True) ? true : false;

	return 0;
}

PyObject * PyC45Tree_get_STOCH_DEBUG(PyC45TreeObject *self, void *closure)
{
	PyObject *res = (self->STOCH_DEBUG) ? Py_True : Py_False;
	Py_INCREF(res);
	return res;
}

int PyC45Tree_set_STOCH_DEBUG(PyC45TreeObject *self, PyObject *value, void *closure)
{
	if (value == NULL) {
		PyErr_SetString(PyExc_TypeError, "Cannot delete the DTDEBUG attribute");
		return -1;
	}

	if (!PyBool_Check(value)) {
		PyErr_SetString(PyExc_TypeError,
			"The out attribute value must be a boolean");
		return -1;
	}

	self->STOCH_DEBUG = (value == Py_True) ? true : false;

	return 0;
}

PyObject * PyC45Tree_get_NODEDEBUG(PyC45TreeObject *self, void *closure)
{
	PyObject *res = (self->NODEDEBUG) ? Py_True : Py_False;
	Py_INCREF(res);
	return res;
}

int PyC45Tree_set_NODEDEBUG(PyC45TreeObject *self, PyObject *value, void *closure)
{
	if (value == NULL) {
		PyErr_SetString(PyExc_TypeError, "Cannot delete the DTDEBUG attribute");
		return -1;
	}

	if (!PyBool_Check(value)) {
		PyErr_SetString(PyExc_TypeError,
			"The out attribute value must be a boolean");
		return -1;
	}

	self->NODEDEBUG = (value == Py_True) ? true : false;

	return 0;
}

static PyGetSetDef PyC45Tree_getseters[] = {
	{ "DTDEBUG",
	(getter)PyC45Tree_get_DTDEBUG, (setter)PyC45Tree_set_DTDEBUG,
	"debug decision tree",
	NULL },
	{ "SPLITDEBUG",
	(getter)PyC45Tree_get_SPLITDEBUG, (setter)PyC45Tree_set_SPLITDEBUG,
	"debug decision tree splits",
	NULL },
	{ "STOCH_DEBUG",
	(getter)PyC45Tree_get_STOCH_DEBUG, (setter)PyC45Tree_set_STOCH_DEBUG,
	"debug decision tree stochastics",
	NULL },
	{ "NODEDEBUG",
	(getter)PyC45Tree_get_NODEDEBUG, (setter)PyC45Tree_set_NODEDEBUG,
	"debug decision tree nodes",
	NULL },
	{ NULL }  /* Sentinel */
};


static bool deleteAll(tree_experience* e) { delete e->input; e->input = NULL; return true; }

void PyC45Tree_dealloc(PyC45TreeObject* self)
{
	delete self->freeNodes;
	self->freeNodes = NULL;

	for (int i = 0; i < N_C45_NODES; ++i) {
		delete self->allNodes[i].outputs;
		self->allNodes[i].outputs = NULL;
	}

	std::remove_if(self->experiences->begin(), self->experiences->end(), deleteAll);
	delete self->experiences;
	self->experiences = NULL;

	Py_XDECREF(self->rng);

	self->ob_type->tp_free((PyObject*)self);
}

PyObject * PyC45Tree_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyC45TreeObject *self;

	self = (PyC45TreeObject *)type->tp_alloc(type, 0);
	if (self != NULL) {
		self->nNodes = 0;
		self->nOutput = 0;
		self->nExperiences = 0;
		self->hadError = false;
		self->maxNodes = N_C45_NODES;
		self->totalNodes = 0;

		self->allow_only_splits = true;

		// how close a split has to be to be randomly selected
		self->SPLIT_MARGIN = 0.0f; //0.02; //5; //01; //0.05; //0.2; //0.05;

		self->MIN_GAIN_RATIO = 0.0001f; //0.0004; //0.001; //0.0002; //0.001;

		self->DTDEBUG = false;
		self->SPLITDEBUG = false;
		self->STOCH_DEBUG = false;
		self->NODEDEBUG = false;

		self->experiences = new std::vector<tree_experience*>();

		for (int i = 0; i < N_C45_NODES; ++i) {
			self->allNodes[i].outputs = new std::map<double, int>();
		}

		self->freeNodes = new std::vector<int>();

		initNodes(self);
		initTree(self);
	}

	return (PyObject *)self;
}

int PyC45Tree_init(PyC45TreeObject *self, PyObject *args, PyObject *kwds)
{
	PyRandomObject *rng = NULL, *tmp;
	bool isnull = false;

	static char *kwlist[] = { "id", "trainMode", "trainFreq", "m", "featPct", "rng", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiifO!", kwlist,
		&self->id,
		&self->mode,
		&self->freq,
		&self->m,
		&self->featPct,
		&PyRandom_Type,
		&rng))
		return -1;

	if (!PyRandom_Check(rng)){
		PyErr_SetString(PyExc_TypeError,
			"rng attribute must be of type Random");
		return -1;
	}

	if (rng) {
		isnull = (self->rng == NULL);

		tmp = self->rng;
		Py_INCREF(rng);
		self->rng = rng;

		if (!isnull) {
			Py_DECREF(tmp);
		}
	}

	cout << "Created C4.5 decision tree " << self->id;
	if (self->allow_only_splits) cout << " with == and > splits" << endl;
	else cout << " with > splits only" << endl;
	if (self->DTDEBUG) {
		cout << " mode: " << self->mode << " freq: " << self->freq << endl;
	}

	return 0;
}

PyObject * PyC45Tree_trainInstance(PyC45TreeObject *self, PyObject *args, PyObject *kwds)
{
	PyClassPairObject *instance = NULL;

	static char *kwlist[] = { "instance", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
		&PyClassPair_Type,
		&instance))
		return NULL;

	if (self->DTDEBUG) cout << "trainInstance" << endl;

	bool modelChanged = false;

	// simply add this instance to the set of experiences

	// take from static array until we run out
	tree_experience *e;
	if (self->nExperiences < N_C45_EXP){
		// from statically created set of experiences
		e = &(self->allExp[self->nExperiences]);
	}
	else {
		// dynamically create experience
		e = new tree_experience;
	}
	// allocate memory for input
	e->input = new std::vector<double>();

	int dim = PyArray_DIM(instance->in_, 0);
	double* in = pyvector_to_Carrayptrs(instance->in_, 1, dim);

	e->input->assign(in, in + dim);
	e->output = instance->out;
	self->experiences->push_back(e);
	self->nExperiences++;

	if (self->nExperiences == 1000000){
		cout << "Reached limit of # experiences allowed." << endl;
		return false;
	}

	if (self->nExperiences != (int)self->experiences->size())
		cout << "ERROR: experience size mismatch: " << self->nExperiences << ", " << self->experiences->size() << endl;

	if (self->DTDEBUG) {
		cout << "Original input: ";
		for (unsigned i = 0; i < (unsigned)dim; i++){
			cout << in[i] << ", ";
		}
		cout << endl << " Original output: " << instance->out << endl;
		cout << "Added exp id: " << self->nExperiences << " output: " << e->output << endl;
		cout << "Address: " << e << " Input : ";
		for (unsigned i = 0; i < e->input->size(); i++){
			cout << (*e->input)[i] << ", ";
		}
		cout << endl << " Now have " << self->nExperiences << " experiences." << endl;
	}

	// depending on mode/etc, maybe re-build tree

	// mode 0: re-build every step
	if (self->mode == BUILD_EVERY || self->nExperiences <= 1) {
		modelChanged = rebuildTree(self);
	}
	// mode 1: re-build on error only
	else if (self->mode == BUILD_ON_ERROR) {

		// build on misclassification
		// check for misclassify
		// get leaf
		tree_node* leaf = traverseTree(self, self->root, (*e->input));
		// find probability for this output
		float count = (float)(*leaf->outputs)[e->output];
		float outputProb = count / (float)leaf->nInstances;

		if (outputProb < 0.75){
			modelChanged = rebuildTree(self);
			modelChanged = true;
		}
	}
	// mode 2: re-build every FREQ steps
	else if (self->mode == BUILD_EVERY_N) {
		// build every freq steps
		if (!modelChanged && (self->nExperiences % self->freq) == 0) {
			modelChanged = rebuildTree(self);
		}
	}

	if (modelChanged) {
		if (self->DTDEBUG) cout << "DT " << self->id << " tree re-built." << endl;

		if (self->DTDEBUG){
			cout << endl << "DT: " << self->id << endl;
			printTree(self->root, 0);
			cout << "Done printing tree" << endl;
		}
	}

	/*
	if (nExperiences % 50 == 0){
	cout << endl << "DT: " << id << endl;
	printTree(root, 0);
	cout << "Done printing tree" << endl;
	}
	*/

	PyObject *res;
	res = (modelChanged) ? Py_True : Py_False;
	Py_INCREF(res);
	return res;
}

PyObject * PyC45Tree_trainInstances(PyC45TreeObject *self, PyObject *args, PyObject *kwds)
{
	PyObject* instances = NULL, *item;

	static char *kwlist[] = { "instances", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
		&PyClassPairList_Type,
		&instances))
		return NULL;

	if (self->DTDEBUG) cout << "trainInstances" << endl;

	bool modelChanged = false;
	bool doBuild = false;
	bool buildOnError = false;

	// loop through instances, possibly checking for errors
	instances = PyObject_GetIter(instances);
	if (instances) {
		while (item = PyIter_Next(instances)) {
			PyClassPairObject* instance = (PyClassPairObject *)item;

			// simply add this instance to the set of experiences

			// take from static array until we run out
			tree_experience *e;
			if (self->nExperiences < N_C45_EXP){
				// from statically created set of experiences
				e = &(self->allExp[self->nExperiences]);
			}
			else {
				// dynamically create experience
				e = new tree_experience;
			}
			// allocate memory for input
			e->input = new std::vector<double>();

			int dim = PyArray_DIM(instance->in_, 0);
			double* in = pyvector_to_Carrayptrs(instance->in_, 1, dim);

			e->input->assign(in, in + dim);
			e->output = instance->out;
			self->experiences->push_back(e);
			self->nExperiences++;

			if (self->nExperiences == 1000000){
				cout << "Reached limit of # experiences allowed." << endl;
				return false;
			}

			if (self->nExperiences != (int)self->experiences->size())
				cout << "ERROR: experience size mismatch: " << self->nExperiences << ", " << self->experiences->size() << endl;

			if (self->DTDEBUG) {
				cout << "Original input: ";
				for (unsigned i = 0; i < (unsigned)dim; i++){
					cout << in[i] << ", ";
				}
				cout << endl << " Original output: " << instance->out << endl;
				cout << "Added exp id: " << self->nExperiences << " output: " << e->output << endl;
				cout << "Address: " << e << " Input : ";
				for (unsigned i = 0; i < e->input->size(); i++){
					cout << (*e->input)[i] << ", ";
				}
				cout << endl << " Now have " << self->nExperiences << " experiences." << endl;
			}

			// depending on mode/etc, maybe re-build tree

			// don't need to check if we've already decided
			if (doBuild) continue;

			// mode 0: re-build every step
			if (self->mode == BUILD_EVERY || self->nExperiences <= 1) {
				doBuild = true;;
			}
			// mode 1: re-build on error only
			else if (self->mode == BUILD_ON_ERROR) {

				// build on misclassification
				// check for misclassify
				// get leaf
				tree_node* leaf = traverseTree(self, self->root, (*e->input));
				// find probability for this output
				float count = (float)(*leaf->outputs)[e->output];
				float outputProb = count / (float)leaf->nInstances;

				if (outputProb < 0.75){
					doBuild = true;
					buildOnError = true;
				}
			}
			// mode 2: re-build every FREQ steps
			else if (self->mode == BUILD_EVERY_N) {
				// build every freq steps
				if ((self->nExperiences % self->freq) == 0) {
					doBuild = true;
				}
			}
			Py_DECREF(item);
		}
		/* clean up */
		Py_DECREF(instances);
	}

	if (self->DTDEBUG) cout << "Added " << PyList_Size(instances) << " new instances. doBuild = " << doBuild << endl;

	if (doBuild){
		modelChanged = rebuildTree(self);
	}

	if (modelChanged){
		if (self->DTDEBUG) cout << "DT " << self->id << " tree re-built." << endl;

		if (self->DTDEBUG){
			cout << endl << "DT: " << self->id << endl;
			printTree(self->root, 0);
			cout << "Done printing tree" << endl;
		}
	}

	PyObject *res;
	res = (modelChanged || buildOnError) ? Py_True : Py_False;
	Py_INCREF(res);
	return res;
}

PyObject * PyC45Tree_testInstance(PyC45TreeObject *self, PyObject *args, PyObject *kwds)
{
	std::vector<double> input;
	PyArrayObject* in_ = NULL;
	PyObject* retval = PyDict_New();

	static char *kwlist[] = { "input", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
		&PyArray_Type,
		&in_))
		return NULL;

	/* Check that object input is 'double' type and a vector */
	if (not_doublevector(in_)) return NULL;

	if (self->DTDEBUG) cout << "testInstance" << endl;

	// in case the tree is empty
	if (self->experiences->size() == 0) {
		PyDict_SetItem(retval, PyFloat_FromDouble(0.0f), PyFloat_FromDouble(1.0));
		return retval;
	}

	int dim = PyArray_DIM(in_, 0);
	double* in = pyvector_to_Carrayptrs(in_, 1, dim);
	input.assign(in, in + dim);

	// follow through tree to leaf
	tree_node* leaf = traverseTree(self, self->root, input);
	self->lastNode = leaf;

	// and return mapping of outputs and their probabilities
	outputProbabilities(self, leaf, retval);

	return retval;
}

PyObject * PyC45Tree_getConf(PyC45TreeObject *self, PyObject *args)
{
	if (self->DTDEBUG) cout << "numVisits" << endl;

	// in case the tree is empty
	if (self->experiences->size() == 0){
		return Py_BuildValue("f", 0.0f);
	}

	// and return # in this leaf
	float conf = (float)self->lastNode->nInstances / (float)(2.0*self->m);
	if (conf > 1.0)
		return Py_BuildValue("f", 1.0f);
	else
		return Py_BuildValue("f", conf);
}

PyObject * PyC45Tree_sortOnDim(PyC45TreeObject *self, PyObject *args, PyObject *kwds)
{
	PyArrayObject *pyvalues;
	PyObject* instances = NULL, *item;
	int dim;

	static char *kwlist[] = { "dim", "instances", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO", kwlist,
		&dim,
		&instances))
		return NULL;

	if (self->DTDEBUG) cout << "sortOnDim,dim = " << dim << endl;

	int len = PyList_Size(instances);
	int dims[2];
	dims[0] = len;
	pyvalues = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);
	double* values = pyvector_to_Carrayptrs(pyvalues, 1, len);

	int i = 0;
	instances = PyObject_GetIter(instances);
	if (instances) {
		while (item = PyIter_Next(instances)) {
			PyClassPairObject* instance = (PyClassPairObject *)item;

			double val = PyFloat_AsDouble(PyList_GetItem((PyObject *)PyArray_ToList(instance->in_), dim));
			//cout << " val: " << val << endl;

			// find where this should go
			for (int j = 0; j <= i; j++){
				//cout << " j: " << j << endl;

				// get to i, this is the spot then
				if (j == i){
					values[j] = val;
					//cout << "  At i, putting value in slot j: " << j << endl;
				}

				// if this is the spot
				else if (val < values[j]){
					//cout << "  Found slot at j: " << j << endl;

					// slide everything forward to make room
					for (int k = i; k > j; k--){
						//cout << "   k = " << k << " Sliding value from k-1 to k" << endl;
						values[k] = values[k - 1];
					}

					// put value in its spot at j
					//cout << "  Putting value at slot j: " << j << endl;
					values[j] = val;

					// break
					break;
				}

			}
			i++;
			Py_DECREF(item);
		}
		/* clean up */
		Py_DECREF(instances);
	}

	if (self->DTDEBUG){
		cout << "Sorted array: " << values[0];
		for (int i = 1; i < len; i++){
			cout << ", " << values[i];
		}
		cout << endl;
	}

	return PyArray_Return(pyvalues);
}

PyObject * PyC45Tree_printTree(PyC45TreeObject *self, PyObject *args, PyObject *kwds)
{
	int level;

	static char *kwlist[] = { "level", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist,
		&level))
		return NULL;

	printTreeAll(self, level);

	Py_RETURN_NONE;
}

static PyMethodDef PyC45Tree_methods[] = {
	{ "train_instance", (PyCFunction)PyC45Tree_trainInstance, METH_VARARGS | METH_KEYWORDS,
	"Train instance. Target ouptut will be a single value." },
	{ "train_instances", (PyCFunction)PyC45Tree_trainInstances, METH_VARARGS | METH_KEYWORDS,
	"Train multiple instances." },
	{ "test_instance", (PyCFunction)PyC45Tree_testInstance, METH_VARARGS | METH_KEYWORDS,
	"Test instance." },
	{ "get_conf", (PyCFunction)PyC45Tree_getConf, METH_VARARGS,
	"Get confidence." },
	{ "sort_on_dim", (PyCFunction)PyC45Tree_sortOnDim, METH_VARARGS | METH_KEYWORDS,
	"Returns an array of the values of features at the index dim, sorted from lowest to highest." },
	{ "print_tree", (PyCFunction)PyC45Tree_printTree, METH_VARARGS | METH_KEYWORDS,
	"Print the tree for debug purposes." },
	{ NULL }  /* Sentinel */
};


PyTypeObject PyC45Tree_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                               /*ob_size*/
	"C45Tree",						 /*tp_name*/
	sizeof(PyC45TreeObject),         /*tp_basicsize*/
	0,                               /*tp_itemsize*/
	(destructor)PyC45Tree_dealloc,   /*tp_dealloc*/
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
	"PyC45Tree objects",             /* tp_doc */
	0,		                         /* tp_traverse */
	0,		                         /* tp_clear */
	0,		                         /* tp_richcompare */
	0,		                         /* tp_weaklistoffset */
	0,		                         /* tp_iter */
	0,		                         /* tp_iternext */
	PyC45Tree_methods,               /* tp_methods */
	PyC45Tree_members,               /* tp_members */
	PyC45Tree_getseters,             /* tp_getset */
	0,                               /* tp_base */
	0,                               /* tp_dict */
	0,                               /* tp_descr_get */
	0,                               /* tp_descr_set */
	0,                               /* tp_dictoffset */
	(initproc)PyC45Tree_init,        /* tp_init */
	0,                               /* tp_alloc */
	PyC45Tree_new,                   /* tp_new */
};



tree_node* allocateNode(PyC45TreeObject *self)
{
	if (self->freeNodes->empty()){
		tree_node* newNode = new tree_node;
		initTreeNode(self, newNode);
		if (self->NODEDEBUG)
			cout << self->id << " PROBLEM: No more pre-allocated nodes!!!" << endl
			<< "return new node " << newNode->id
			<< ", now " << self->freeNodes->size() << " free nodes." << endl;
		return newNode;
	}

	int i = self->freeNodes->back();
	self->freeNodes->pop_back();
	if (self->NODEDEBUG)
		cout << self->id << " allocate node " << i << " with id " << self->allNodes[i].id
		<< ", now " << self->freeNodes->size() << " free nodes." << endl;
	return &(self->allNodes[i]);
}

void deallocateNode(PyC45TreeObject *self, tree_node* node)
{

	if (node->id >= N_C45_NODES){
		if (self->NODEDEBUG)
			cout << self->id << " dealloc extra node id " << node->id
			<< ", now " << self->freeNodes->size() << " free nodes." << endl;
		delete node;
		return;
	}

	self->freeNodes->push_back(node->id);
	if (self->NODEDEBUG)
		cout << self->id << " dealloc node " << node->id
		<< ", now " << self->freeNodes->size() << " free nodes." << endl;
}

// init the tree
void initTree(PyC45TreeObject *self)
{
	if (self->DTDEBUG) cout << "initTree()" << endl;
	self->root = allocateNode(self);

	if (self->DTDEBUG) cout << "   root id = " << self->root->id << endl;

	// just to ensure the diff models are on different random values
	for (int i = 0; i < self->id; i++){
		uniform(self->rng, 0, 1);
	}

}

// init a tree node
void initTreeNode(PyC45TreeObject *self, tree_node *node)
{
	if (self->DTDEBUG) cout << "initTreeNode()";

	node->id = self->nNodes++;
	if (self->DTDEBUG) cout << " id = " << node->id << endl;

	self->totalNodes++;
	if (self->totalNodes > self->maxNodes){
		self->maxNodes = self->totalNodes;
		if (self->DTDEBUG) cout << self->id << " C4.5 MAX nodes: " << self->maxNodes << endl;
	}

	// split criterion
	node->dim = -1;
	node->val = -1;
	node->type = INVALID;

	// current data
	node->nInstances = 0;
	node->outputs->clear();

	// next nodes in tree
	node->l = NULL;
	node->r = NULL;

	node->leaf = true;
}


void initNodes(PyC45TreeObject *self)
{
	for (int i = 0; i < N_C45_NODES; i++){
		initTreeNode(self, &(self->allNodes[i]));
		self->freeNodes->push_back(i);
		if (self->NODEDEBUG)
			cout << "init node " << i << " with id " << self->allNodes[i].id
			<< ", now " << self->freeNodes->size() << " free nodes." << endl;
	}
}


void deleteTree(PyC45TreeObject* self, tree_node* node)
{
	if (self->DTDEBUG) cout << "deleteTree, node=" << node->id << endl;

	if (node == NULL)
		return;

	self->totalNodes--;

	node->nInstances = 0;
	node->outputs->clear();

	//recursively call deleteTree on children
	// then delete them
	if (!node->leaf){
		// left child
		if (node->l != NULL){
			deleteTree(self, node->l);
			deallocateNode(self, node->l);
			node->l = NULL;
		}

		// right child
		if (node->r != NULL){
			deleteTree(self, node->r);
			deallocateNode(self, node->r);
			node->r = NULL;
		}
	}

	node->leaf = true;
	node->dim = -1;

}

bool makeLeaf(PyC45TreeObject* self, tree_node* node)
{

	// check on children
	if (node->l != NULL){
		deleteTree(self, node->l);
		deallocateNode(self, node->l);
		node->l = NULL;
	}

	if (node->r != NULL){
		deleteTree(self, node->r);
		deallocateNode(self, node->r);
		node->r = NULL;
	}

	// changed from not leaf to leaf, or just init'd
	bool change = (!node->leaf || node->type < 0);

	node->leaf = true;
	node->type = ONLY;

	return change;
}

bool implementSplit(PyC45TreeObject* self, tree_node* node, float bestGainRatio, int bestDim,
	float bestVal, splitTypes bestType,
	const std::vector<tree_experience*> &bestLeft,
	const std::vector<tree_experience*> &bestRight,
	bool changed)
{
	if (self->DTDEBUG) cout << "implementSplit node=" << node->id << ",gainRatio=" << bestGainRatio
		<< ",dim=" << bestDim
		<< ",val=" << bestVal << ",type=" << bestType
		<< ",chg=" << changed << endl;


	// see if this should still be a leaf node
	if (bestGainRatio < self->MIN_GAIN_RATIO){
		bool change = makeLeaf(self, node);
		if (self->SPLITDEBUG || self->STOCH_DEBUG){
			cout << "DT " << self->id << " Node " << node->id << " Poor gain ratio: "
				<< bestGainRatio << ", " << node->nInstances
				<< " instances classified at leaf " << node->id
				<< " with multiple outputs " << endl;
		}
		return change;
	}

	// see if this split changed or not
	// assuming no changes above
	if (!changed && node->dim == bestDim && node->val == bestVal
		&& node->type == bestType && !node->leaf
		&& node->l != NULL && node->r != NULL){
		// same split as before.
		if (self->DTDEBUG || self->SPLITDEBUG) cout << "Same split as before" << endl;
		bool changeL = false;
		bool changeR = false;

		// see which leaf changed
		if (bestLeft.size() > (unsigned)node->l->nInstances){
			// redo left side
			if (self->DTDEBUG) cout << "Rebuild left side of tree" << endl;
			changeL = buildTree(self, node->l, bestLeft, changed);
		}

		if (bestRight.size() > (unsigned)node->r->nInstances){
			// redo right side
			if (self->DTDEBUG) cout << "Rebuild right side of tree" << endl;
			changeR = buildTree(self, node->r, bestRight, changed);
		}

		// everything up to here is the same, check if there were changes below
		return (changeL || changeR);
	}

	// totally new
	// set the best split here
	node->leaf = false;
	node->dim = bestDim;
	node->val = bestVal;
	node->type = bestType;

	if (self->SPLITDEBUG) cout << "Best split was type " << node->type
		<< " with val " << node->val
		<< " on dim " << node->dim
		<< " with gainratio: " << bestGainRatio << endl;

	if (self->DTDEBUG) cout << "Left has " << bestLeft.size()
		<< ", right has " << bestRight.size() << endl;

	// make sure both instances
	if (bestLeft.size() == 0 || bestRight.size() == 0){
		cout << "ERROR: DT " << self->id << " node " << node->id << " has 0 instances: left: " << bestLeft.size()
			<< " right: " << bestRight.size() << endl;
		cout << "Split was type " << node->type
			<< " with val " << node->val
			<< " on dim " << node->dim
			<< " with gainratio: " << bestGainRatio << endl;
		exit(-1);
	}


	// check if these already exist
	if (node->l == NULL){
		if (self->DTDEBUG) cout << "Init new left tree nodes " << endl;
		node->l = allocateNode(self);
	}
	if (node->r == NULL){
		if (self->DTDEBUG) cout << "Init new right tree nodes " << endl;
		node->r = allocateNode(self);
	}

	// recursively build the sub-trees to this one
	if (self->DTDEBUG) cout << "Building left tree for node " << node->id << endl;
	buildTree(self, node->l, bestLeft, true);
	if (self->DTDEBUG) cout << "Building right tree for node " << node->id << endl;
	buildTree(self, node->r, bestRight, true);

	// this one changed, or above changed, no reason to check change of lower parts
	return true;
}

void compareSplits(PyC45TreeObject* self, float gainRatio, int dim, float val, splitTypes type,
	const std::vector<tree_experience*> &left,
	const std::vector<tree_experience*> &right,
	int *nties, float *bestGainRatio, int *bestDim,
	float *bestVal, splitTypes *bestType,
	std::vector<tree_experience*> *bestLeft, std::vector<tree_experience*> *bestRight)
{
	if (self->DTDEBUG) cout << "compareSplits gainRatio=" << gainRatio << ",dim=" << dim
		<< ",val=" << val << ",type= " << type << endl;


	bool newBest = false;

	// if its a virtual tie, break it randomly
	if (fabs(*bestGainRatio - gainRatio) < self->SPLIT_MARGIN){
		//cout << "Split tie, making random decision" << endl;

		(*nties)++;
		float randomval = uniform(self->rng, 0, 1);
		float newsplitprob = (1.0f / (float)*nties);

		if (randomval < newsplitprob){
			newBest = true;
			if (self->SPLITDEBUG) cout << "   Tie on split. DT: " << self->id << " rand: " << randomval
				<< " splitProb: " << newsplitprob << ", selecting new split " << endl;
		}
		else
			if (self->SPLITDEBUG) cout << "   Tie on split. DT: " << self->id << " rand: " << randomval
				<< " splitProb: " << newsplitprob << ", staying with old split " << endl;
	}

	// if its clearly better, set this as the best split
	else if (gainRatio > *bestGainRatio){
		newBest = true;
		*nties = 1;
	}


	// set the split features
	if (newBest){
		*bestGainRatio = gainRatio;
		*bestDim = dim;
		*bestVal = val;
		*bestType = type;
		*bestLeft = left;
		*bestRight = right;
		if (self->SPLITDEBUG){
			cout << "  New best gain ratio: " << *bestGainRatio
				<< ": type " << *bestType
				<< " with val " << *bestVal
				<< " on dim " << *bestDim << endl;
		}
	} // newbest
}

void testPossibleSplits(PyC45TreeObject* self, 
	const std::vector<tree_experience*> &instances,
	float *bestGainRatio, int *bestDim,
	float *bestVal, splitTypes *bestType,
	std::vector<tree_experience*> *bestLeft,
	std::vector<tree_experience*> *bestRight)
{
	if (self->DTDEBUG) cout << "testPossibleSplits" << endl;


	// pre-calculate some stuff for these splits (namely I, P, C)
	float I = calcIforSet(self, instances);
	//if (DTDEBUG) cout << "I: " << I << endl;

	int nties = 0;

	// for each possible split, calc gain ratio
	for (unsigned idim = 0; idim < instances[0]->input->size(); idim++){

		//float* sorted = sortOnDim(idim, instances);
		double minVal, maxVal;
		std::set<double> uniques = getUniques(self, idim, instances, minVal, maxVal);

		for (std::set<double>::iterator j = uniques.begin(); j != uniques.end(); j++){

			// skip max val, not a valid cut for either
			if ((*j) == maxVal)
				continue;

			// if this is a random forest, we eliminate some random number of splits
			// here (decision is taken from the random set that are left)
			if (uniform(self->rng) < self->featPct)
				continue;

			std::vector<tree_experience*> left;
			std::vector<tree_experience*> right;

			// splits that are cuts
			float splitval = (float)(*j);
			float gainRatio = calcGainRatio(self, idim, splitval, CUT, instances, I, left, right);

			if (self->SPLITDEBUG) cout << " CUT split val " << splitval
				<< " on dim: " << idim << " had gain ratio "
				<< gainRatio << endl;

			// see if this is the new best gain ratio
			compareSplits(self, gainRatio, idim, splitval, CUT, left, right, &nties,
				bestGainRatio, bestDim, bestVal, bestType, bestLeft, bestRight);


			// no minval here, it would be the same as the cut split on minval
			if (self->allow_only_splits && (*j) != minVal){
				// splits that are true only if this value is equal
				float splitval = (float)(*j);

				float gainRatio = calcGainRatio(self, idim, splitval, ONLY, instances, I, left, right);

				if (self->SPLITDEBUG) cout << " ONLY split val " << splitval
					<< " on dim: " << idim << " had gain ratio "
					<< gainRatio << endl;

				// see if this is the new best gain ratio
				compareSplits(self, gainRatio, idim, splitval, ONLY, left, right, &nties,
					bestGainRatio, bestDim, bestVal, bestType, bestLeft, bestRight);

			} // splits with only

		} // j loop
	}
}

bool buildTree(PyC45TreeObject* self, 
	tree_node *node,
	const std::vector<tree_experience*> &instances,
	bool changed)
{
	if (self->DTDEBUG) cout << "buildTree, node=" << node->id
		<< ",nInstances:" << instances.size()
		<< ",chg:" << changed << endl;

	if (instances.size() == 0){
		cout << "Error: buildTree called on tree " << self->id << " node " << node->id << " with no instances." << endl;
		exit(-1);
	}


	// TODO: what about stochastic data?
	//std::vector<float> chiSquare = calcChiSquare(instances);

	// first, add instances to tree
	node->nInstances = instances.size();

	// add each output to this node
	node->outputs->clear();
	for (unsigned i = 0; i < instances.size(); i++){
		(*node->outputs)[instances[i]->output]++;
	}

	// see if they're all the same
	if (node->outputs->size() == 1){
		bool change = makeLeaf(self, node);
		if (self->DTDEBUG){
			cout << "All " << node->nInstances
				<< " classified with output "
				<< instances[0]->output << endl;
		}
		return change;
	}

	// if not, calculate gain ratio to determine best split
	else {

		if (self->SPLITDEBUG) cout << endl << "Creating new decision node" << endl;

		//node->leaf = false;
		//node->nInstances++;

		float bestGainRatio = -1.0;
		int bestDim = -1;
		float bestVal = -1;
		splitTypes bestType = ONLY;
		std::vector<tree_experience*> bestLeft;
		std::vector<tree_experience*> bestRight;

		testPossibleSplits(self, instances, &bestGainRatio, &bestDim, &bestVal, &bestType, &bestLeft, &bestRight);

		return implementSplit(self, node, bestGainRatio, bestDim, bestVal, bestType, bestLeft, bestRight, changed);

	}
}

bool rebuildTree(PyC45TreeObject* self)
{
	return buildTree(self, self->root, *self->experiences, false);
}


tree_node* getCorrectChild(PyC45TreeObject* self, tree_node* node, const std::vector<double> &input)
{

	if (self->DTDEBUG) cout << "getCorrectChild, node=" << node->id << endl;

	if (passTest(self, node->dim, node->val, node->type, input))
		return node->l;
	else
		return node->r;

}

tree_node* traverseTree(PyC45TreeObject* self, tree_node* node, const std::vector<double> &input)
{

	if (self->DTDEBUG) cout << "traverseTree, node=" << node->id << endl;

	while (!node->leaf){
		node = getCorrectChild(self, node, input);
	}

	if (self->DTDEBUG) cout << "traverseTree, child=" << node->id << endl;

	return node;
}


// output a map of outcomes and their probabilities for this leaf node
void outputProbabilities(PyC45TreeObject* self, tree_node* leaf, PyObject* retval)
{
	if (self->STOCH_DEBUG) cout << "Calculating output probs for leaf " << leaf->id << endl;

	// go through all output values at this leaf, turn into probabilities
	for (std::map<double, int>::iterator it = leaf->outputs->begin();
		it != leaf->outputs->end(); it++) {

		double val = (*it).first;
		float count = (float)(*it).second;
		if (count > 0)
			PyDict_SetItem(retval, PyFloat_FromDouble(val), PyFloat_FromDouble(count / (float)leaf->nInstances));

		if (self->STOCH_DEBUG)
			cout << "Output value " << val << " had count of " << count << " on "
			<< leaf->nInstances << " instances and prob of "
			<< PyFloat_AsDouble(PyDict_GetItem(retval, PyFloat_FromDouble(val))) << endl;
	}
}

bool passTest(PyC45TreeObject* self, int dim, float val, splitTypes type, const std::vector<double> &input)
{
	if (self->DTDEBUG) cout << "passTest, dim=" << dim << ",val=" << val << ",type=" << type
		<< ",input[" << dim << "]=" << input[dim] << endl;

	if (type == CUT){
		if (input[dim] > val)
			return false;
		else
			return true;
	}
	else if (type == ONLY){
		if (input[dim] == val)
			return false;
		else
			return true;
	}
	else {
		return false;
	}

}

float calcGainRatio(PyC45TreeObject* self, int dim, float val, splitTypes type,
	const std::vector<tree_experience*> &instances,
	float I,
	std::vector<tree_experience*> &left,
	std::vector<tree_experience*> &right)
{
	if (self->DTDEBUG) cout << "calcGainRatio, dim=" << dim
		<< " val=" << val
		<< " I=" << I
		<< " nInstances= " << instances.size() << endl;

	left.clear();
	right.clear();

	// array with percentage positive and negative for this test
	float D[2];

	// info(T) = I(P): float I;

	// Info for this split = Info(X,T)
	float Info;

	// Gain for this split = Gain(X,T)
	float Gain;

	// SplitInfo for this split = I(|pos|/|T|, |neg|/|T|)
	float SplitInfo;

	// GainRatio for this split = GainRatio(X,T) = Gain(X,T) / SplitInfo(X,T)
	float GainRatio;

	// see where the instances would go with this split
	for (unsigned i = 0; i < instances.size(); i++){
		if (self->DTDEBUG) cout << "calcGainRatio - Classify instance " << i
			<< " on new split " << endl;

		if (passTest(self, dim, val, type, *instances[i]->input)){
			left.push_back(instances[i]);
		}
		else{
			right.push_back(instances[i]);
		}
	}

	if (self->DTDEBUG) cout << "Left has " << left.size()
		<< ", right has " << right.size() << endl;

	D[0] = (float)left.size() / (float)instances.size();
	D[1] = (float)right.size() / (float)instances.size();
	float leftInfo = calcIforSet(self, left);
	float rightInfo = calcIforSet(self, right);
	Info = D[0] * leftInfo + D[1] * rightInfo;
	Gain = I - Info;
	SplitInfo = calcIofP(self, (float*)&D, 2);
	GainRatio = Gain / SplitInfo;

	if (self->DTDEBUG){
		cout << "LeftInfo: " << leftInfo
			<< " RightInfo: " << rightInfo
			<< " Info: " << Info
			<< " Gain: " << Gain
			<< " SplitInfo: " << SplitInfo
			<< " GainRatio: " << GainRatio
			<< endl;
	}

	return GainRatio;

}

float calcIofP(PyC45TreeObject* self, float* P, int size)
{
	if (self->DTDEBUG) cout << "calcIofP, size=" << size << endl;
	float I = 0;
	for (int i = 0; i < size; i++){
		I -= P[i] * log(P[i]);
	}
	return I;
}

float calcIforSet(PyC45TreeObject* self, const std::vector<tree_experience*> &instances)
{
	if (self->DTDEBUG) cout << "calcIforSet" << endl;

	std::map<double, int> classes;

	// go through instances and figure count of each type
	for (unsigned i = 0; i < instances.size(); i++){
		// increment count for this value
		double val = instances[i]->output;
		classes[val]++;
	}

	// now calculate P
	float Pval;
	float I = 0;
	for (std::map<double, int>::iterator i = classes.begin(); i != classes.end(); i++){
		Pval = (float)(*i).second / (float)instances.size();
		// calc I of P
		I -= Pval * log(Pval);
	}

	return I;

}

std::set<double> getUniques(PyC45TreeObject* self,
	int dim,
	const std::vector<tree_experience*> &instances,
	double& minVal,
	double& maxVal)
{
	if (self->DTDEBUG) cout << "getUniques,dim = " << dim;

	std::set<double> uniques;

	for (int i = 0; i < (int)instances.size(); i++){
		if (i == 0 || (*instances[i]->input)[dim] < minVal)
			minVal = (*instances[i]->input)[dim];
		if (i == 0 || (*instances[i]->input)[dim] > maxVal)
			maxVal = (*instances[i]->input)[dim];

		uniques.insert((*instances[i]->input)[dim]);
	}

	if (self->DTDEBUG) cout << " #: " << uniques.size() << endl;
	return uniques;
}


void printTreeAll(PyC45TreeObject* self, int level)
{
	cout << endl << "DT: " << self->id << endl;
	printTree(self->root, level);
	cout << "Done printing tree" << endl;
}

void printTree(tree_node *t, int level)
{

	for (int i = 0; i < level; i++){
		cout << ".";
	}

	cout << "Node " << t->id;
	if (t->type == CUT) cout << " Type: CUT";
	else                cout << " Type: ONLY";
	cout << " Dim: " << t->dim << " Val: " << t->val
		<< " nInstances: " << t->nInstances;

	if (t->leaf){
		cout << " Outputs: ";
		for (std::map<double, int>::iterator j = t->outputs->begin();
			j != t->outputs->end(); j++){
			cout << (*j).first << ": " << (*j).second << ", ";
		}
		cout << endl;
	}
	else
		cout << " Left: " << t->l->id << " Right: " << t->r->id << endl;


	// print children
	if (t->dim != -1 && !t->leaf){
		printTree(t->l, level + 1);
		printTree(t->r, level + 1);
	}

}

