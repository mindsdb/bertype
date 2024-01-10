// disable deprecated numpy API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL __bertype_column_to_sentences_extension__
#define NO_IMPORT_ARRAY

#include <stdio.h>
#include <string.h>
#include <math.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "col2sent.h"

PyObject* column_to_sentences(PyObject* self, PyObject* args)
{
  int err;

  npy_short c;
  npy_short lc;
  npy_short* buffer;
  npy_short* data_str;
  npy_int64* sentence_lengths;

  // counters
  // - character position in global input buffer
  npy_int64 i;
  // - character position in stride
  npy_int64 j;
  // - sentence counter
  npy_int64 m;
  // number of characters in input buffer
  npy_int64 N;
  // maximum number of sentences
  npy_int64 M;
  // maximum allowed length of sentences
  npy_int64 max_length;
  // minimum allowed length of sentences
  npy_int64 min_length;
  // signals state of the main loop
  npy_short flag;

  PyObject* py_data_str;
  PyObject* py_sentence_lengths;
  PyArrayObject* pyarr_data_str;
  PyArrayObject* pyarr_sentence_lengths;

  if (!PyArg_ParseTuple(args, "lllOO",
                        &N,
                        &max_length,
                        &min_length,
                        &py_data_str,
                        &py_sentence_lengths))
  {
    return NULL;
  }

  // debug prints
  printf("N is %ld\n", N);
  printf("max_length is %ld\n", max_length);
  printf("min_length is %ld\n", min_length);

  // access input data as C-contiguous array
  pyarr_data_str = (PyArrayObject*)PyArray_FROM_O(py_data_str);
  if (pyarr_data_str == NULL) {
    PyErr_SetString(
      PyExc_ValueError,
      "[col2sent::ERROR] Failed to parse input as PyArrayObject."
    );
    return NULL;
  }
  data_str = (npy_short* )PyArray_DATA(pyarr_data_str);

  // access output as C-contiguous array
  pyarr_sentence_lengths = (PyArrayObject*)PyArray_FROM_OTF(py_sentence_lengths, NPY_INT64, NPY_ARRAY_OUT_ARRAY);
  if (pyarr_sentence_lengths == NULL) {
    PyErr_SetString(
      PyExc_ValueError,
      "[col2sent::ERROR] Failed to parse output as PyArrayObject."
    );
    return NULL;
  }
  sentence_lengths = (npy_int64* )PyArray_DATA(pyarr_sentence_lengths);

  // allocate single-sentence buffer
  buffer = (npy_short*)malloc(sizeof(npy_short) * max_length);

  // initialize counters
  i = 0;
  j = 0;
  m = 0;
  // start loop in continue mode
  flag = C2S_CONTINUE;
  while(i < N)
  {
    // pack sentence
    if(flag == C2S_PACK_SENTENCE)
    {
      //printf("PACKAGING!\n");
      // avoid abrupt truncation by enforcing the last character to be ' '
      // this might fail in cases where a malicious user has columns with
      // entries longer than `max_length` characters, like
      //
      // .  543.38949393939393930109132904873948 (etc)
      //
      // For those cases, a minimum sentence length of 16 is enforced.
      lc = 0; // buffer[j - 1];
      while(lc != ' ' && j >= C2S_MINIMUM_SENTENCE_LENGTH)
      {
        // rewind in-buffer position
        j = j - 1;
        // we shall also rewind the global position, so that the next
        // sentence doesn't start from a truncated word. While this is
        // most certainly not the desired behavior by default, I am
        // leaving a pre-compiler directive you might undefine in case
        // truncated word startings are what you want.
        #ifdef C2S_ALLOW_TRUNCATED_WORD_START
        i = i - 1;
        #endif
        lc = buffer[j];
      }
      // store sentence lenght
      sentence_lengths[m] = j;
      if(m > 0)
      {
        sentence_lengths[m] += sentence_lengths[m - 1];
      }
      //printf("j = %ld m = %ld M = %ld\n", j, m, M);
      // update sentence counter
      m = m + 1;
      // reset in-buffer position
      j = 0;
      // reset flag
      flag = C2S_CONTINUE;
    }
    else if(flag == C2S_CONTINUE)
    {
      // read single character. note the '2*i'; that's some numpy
      // business going on when packing strings into arrays
      c = data_str[2 * i];
      // add to buffer
      buffer[j] = c;
      // update in-buffer position
      j = j + 1;
      // update global-buffer position
      i = i + 1;
    }
    // when end of buffer is reached, trigger packaging
    // only triggers if flag is already in CONTINUE state.
    if(j == max_length && flag == C2S_CONTINUE)
    {
      flag = C2S_PACK_SENTENCE;
    }
  }
  // free temporal buffer
  // free(buffer);
  // printf("PACKAGING DONE\n");
  // set base object to self for correct memory management
  // err = PyArray_SetBaseObject(pyarr_sentence_lengths, self);
  // if(err != 0)
  // {
  //   PyErr_SetString(
  //     PyExc_ValueError,
  //     "[col2sent::ERROR] Could not set base reference for sentence_lengths array."
  //   );
  //   return NULL;
  // }
  // printf("PYTHON OBJECT CAN STEAL REFERENCE FROM SELF\n");
  printf("DONE\n");

  return pyarr_sentence_lengths;
}
