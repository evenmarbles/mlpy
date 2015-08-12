#ifndef C45TREE_H
#define C45TREE_H

#include <vector>
#include <map>
#include <set>

#include "Python.h"
#include "structmember.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_Classifier
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"

#include "random.h"

#define N_C45_EXP 200000
#define N_C45_NODES 2500

#define BUILD_EVERY 0
#define BUILD_ON_ERROR 1
#define BUILD_EVERY_N 2
#define BUILD_ON_TERMINAL 3
#define BUILD_ON_TERMINAL_AND_ERROR 4


/** The types of splits. Split on ONLY meaning is input == x, or CUT meaning is input > x */
enum splitTypes{
	INVALID = -1,
	ONLY,
	CUT
};


/*
* PyTreeNode_Type
*/

/** Tree node struct. For decision nodes, it contains split information and pointers to child nodes. 
For leaf nodes, it contains all outputs that went into this leaf during trainiing. */
typedef struct PyTreeNode PyTreeNodeObject;

typedef struct PyTreeNode {
	PyObject_HEAD
	int id;

	// split criterion
	int dim;
	float val;
	splitTypes type;

	// set of all outputs seen at this leaf/node
	PyDictObject *outputs;
	int nInstance;

	// next nodes in the tree
	PyTreeNodeObject *l;
	PyTreeNodeObject *r;

	bool leaf;
} PyTreeNodeObject;

extern PyTypeObject PyTreeNode_Type;

#define PyTreeNode_Check(op) PyObject_TypeCheck(op, &PyTreeNode_Type)
#define PyTreeNode_CheckExact(op) (Py_TYPE(op) == &PyTreeNode_Type)


/*
* PyTreeExperience_Type
*/
typedef struct _treeexperienceobject PyTreeExperienceObject;
struct _treeexperienceobject {
	PyObject_HEAD
	PyArrayObject *in_; /* tree input */
	double out;	/* tree output */
};

extern PyTypeObject PyTreeExperience_Type;

#define PyTreeExperience_Check(op) PyObject_TypeCheck(op, &PyTreeExperience_Type)
#define PyTreeExperience_CheckExact(op) (Py_TYPE(op) == &PyTreeExperience_Type)



/*
* PyTreeExperienceList_Type
*/
typedef struct _treeexperiencelistobject PyTreeExperienceListObject;
struct _treeexperiencelistobject {
	PyListObject list;
};

extern PyTypeObject PyTreeExperienceList_Type;

#define PyTreeExperienceList_Check(op) PyObject_TypeCheck(op, &PyTreeExperienceList_Type)
#define PyTreeExperienceList_CheckExact(op) (Py_TYPE(op) == &PyTreeExperienceList_Type)



/*
* PyC45Tree_Type
*/

/** Tree node struct. For decision nodes, it contains split information and pointers to child nodes. For leaf nodes, it contains all outputs that went into this leaf during trainiing. */
struct tree_node {
	int id;

	// split criterion
	int dim;
	float val;
	splitTypes type;

	// set of all outputs seen at this leaf/node
	std::map<double, int> *outputs;
	int nInstances;

	// next nodes in tree
	tree_node *l;
	tree_node *r;

	bool leaf;
};

/** Experiences the tree is trained on. A vector of inputs and one float output to predict */
struct tree_experience {
	std::vector<double>* input;
	double output;
};

typedef struct {
	PyObject_HEAD
	int id;
	int mode;
	int freq;
	int m;
	float featPct;
	bool allow_only_splits;
	PyRandomObject *rng;

	int nOutput;
	int nNodes;
	bool hadError;
	int maxNodes;
	int totalNodes;

	int nExperiences;

	bool DTDEBUG;
	bool SPLITDEBUG;
	bool STOCH_DEBUG;
	bool NODEDEBUG;

	float SPLIT_MARGIN;
	float MIN_GAIN_RATIO;

	/** Vector of all experiences used to train the tree */
	std::vector<tree_experience*>* experiences;

	/** Pre-allocated array of experiences to be filled during training. */
	tree_experience allExp[N_C45_EXP];

	/** Pre-allocated array of tree nodes to be used for tree */
	PyTreeNode _allNodes[N_C45_NODES];
	tree_node allNodes[N_C45_NODES];
	std::vector<int>* freeNodes;

	// TREE
	/** Pointer to root node of tree. */
	tree_node* root;
	/** Pointer to last node of tree used (leaf used in last prediction made). */
	tree_node* lastNode;
} PyC45TreeObject;

extern PyTypeObject PyC45Tree_Type;

#define PyC45Tree_Check(op) PyObject_TypeCheck(op, &PyC45Tree_Type)
#define PyC45Tree_CheckExact(op) (Py_TYPE(op) == &PyC45Tree_Type)


// utility functions
tree_node* allocateNode(PyC45TreeObject *self);
void deallocateNode(PyC45TreeObject *self, tree_node* node);
void initTree(PyC45TreeObject *self);
void initTreeNode(PyC45TreeObject *self, tree_node *node);
void initNodes(PyC45TreeObject *self);

void deleteTree(PyC45TreeObject* self, tree_node* node);
bool makeLeaf(PyC45TreeObject* self, tree_node* node);
bool implementSplit(PyC45TreeObject* self,
	tree_node* node, float bestGainRatio, int bestDim,
	float bestVal, splitTypes bestType,
	const std::vector<tree_experience*> &bestLeft,
	const std::vector<tree_experience*> &bestRight,
	bool changed);
void compareSplits(PyC45TreeObject* self, float gainRatio, int dim, float val, splitTypes type,
	const std::vector<tree_experience*> &left,
	const std::vector<tree_experience*> &right,
	int *nties, float *bestGainRatio, int *bestDim,
	float *bestVal, splitTypes *bestType,
	std::vector<tree_experience*> *bestLeft, std::vector<tree_experience*> *bestRight);
void testPossibleSplits(PyC45TreeObject* self,
	const std::vector<tree_experience*> &instances,
	float *bestGainRatio, int *bestDim,
	float *bestVal, splitTypes *bestType,
	std::vector<tree_experience*> *bestLeft,
	std::vector<tree_experience*> *bestRight);

bool buildTree(PyC45TreeObject* self, tree_node *node, const std::vector<tree_experience*> &instances, bool changed);
bool rebuildTree(PyC45TreeObject* self);

tree_node* getCorrectChild(PyC45TreeObject* self, tree_node* node, const std::vector<double> &input);
tree_node* traverseTree(PyC45TreeObject* self, tree_node* node, const std::vector<double> &input);

void outputProbabilities(PyC45TreeObject* self, tree_node* leaf, PyObject* retval);

bool passTest(PyC45TreeObject* self, int dim, float val, splitTypes type, const std::vector<double> &input);
float calcGainRatio(PyC45TreeObject* self, int dim, float val, splitTypes type,
	const std::vector<tree_experience*> &instances,
	float I,
	std::vector<tree_experience*> &left,
	std::vector<tree_experience*> &right);
float calcIofP(PyC45TreeObject* self, float* P, int size);
float calcIforSet(PyC45TreeObject* self, const std::vector<tree_experience*> &instances);
std::set<double> getUniques(PyC45TreeObject* self, int dim, const std::vector<tree_experience*> &instances, double& minVal, double& maxVal);

void printTreeAll(PyC45TreeObject* self, int level);
void printTree(tree_node *t, int level);

#endif		// C45TREE_H
