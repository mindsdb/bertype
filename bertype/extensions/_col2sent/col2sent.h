#ifndef __COL2SEQ_H__
#define __COL2SEQ_H__

// control flags
#define C2S_CONTINUE 0
#define C2S_PACK_SENTENCE -1
#define C2S_MINIMUM_SENTENCE_LENGTH 16
#define C2S_ALLOW_TRUNCATED_WORD_START

#include <Python.h>

PyObject* column_to_sentences(PyObject*, PyObject*);

#endif