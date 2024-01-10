#include "col2sent.h"

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
	"col2sent",
	mod_col2sent_docs,
	-1,
	col2sent_funcs,
	NULL,
	NULL,
	NULL,
	NULL
};
// function to initialize module
// must be called 'PyInit_<name of module>' or it will just not work
// good job Python
PyMODINIT_FUNC PyInit_col2sent(void) {
	return PyModule_Create(&col2sent_mod);
}