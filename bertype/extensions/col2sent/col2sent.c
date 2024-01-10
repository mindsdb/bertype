#include <stdio.h>
#include <Python.h>
#include "col2sent.h"

PyObject* column_to_sentences(PyObject* self, PyObject* args)
{
  char* col_data_str;

  if(!PyArg_ParseTuple(args, "s", &col_data_str))
	  return NULL;

  return PyUnicode_FromFormat("echo: %s", col_data_str);
	// return Py_BuildValue("is", num1 + num2, eq);
}
