""" test_col2sent.py

    Simple script to test the `col2sent` extension.
"""
import numpy
import bertype
print(bertype.__file__)
from bertype.extensions._col2sent import column_to_sentences
import time

tic = time.time()
data = str(numpy.random.randn(100000).tolist())
data = data.replace(',', '')
data_np = numpy.asarray(list(data))
output_np = numpy.zeros(data_np.size // 16 + 1, dtype=numpy.int64)
print(data_np.size)
output_np = column_to_sentences(data_np.size, 64, 16, data_np, output_np)
# mask out entries that are not used
output_np = output_np[output_np > 0]
toc = time.time()
elapsed = toc - tic
s = 0
for e in output_np:
    x = data_np[s:e]
    s = e

print(f'wall time: {elapsed:2.4f}')
