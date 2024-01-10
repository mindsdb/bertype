// disable deprecated numpy API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL __bertype_column_to_sentences_extension__
#include "col2sent.h"
#include <numpy/arrayobject.h>

// documentation for Python-level API of function column_to_sentences
char func_column_to_sentences_docs[] = "Transforms a string representing a column to sentences.";
// define and initialize a Python Method called 'column_to_sentences' that calls the C routine.
PyMethodDef col2sent_funcs[] = {
	{	"column_to_sentences",
		(PyCFunction)column_to_sentences,
		// function receives positional arguments (METH_VARARGS)
		METH_VARARGS,
		func_column_to_sentences_docs},
	// centinel value
	{	NULL}
};
// documentation for Python-level API of *module* col2sent
char mod_col2sent_docs[] = "This is col2sent module.";
// define and initialiaze *module* col2sent
PyModuleDef col2sent_mod = {
	PyModuleDef_HEAD_INIT,
	"_col2sent",
	mod_col2sent_docs,
	-1,
	col2sent_funcs,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__col2sent(void)
{
	import_array();
	return PyModule_Create(&col2sent_mod);
}
