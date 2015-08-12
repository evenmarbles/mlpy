/* Header for Hidden Markov model methods: hmmcmodule.c */

/* ==== Prototypes =================================== */

/* .... Python callable HMM functions ..................*/
static PyObject * forward_wrapper(PyObject * self, PyObject * args);
static PyObject * backward_wrapper(PyObject * self, PyObject * args);
static PyObject * fwdbkw_wrapper(PyObject * self, PyObject * args);
static PyObject * compueTwoSliceSum_wrapper(PyObject * self, PyObject * args);
static PyObject * viterbi_wrapper(PyObject * self, PyObject * args);

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
