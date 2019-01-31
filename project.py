import idx2numpy
import numpy as np

read_array = idx2numpy.convert_from_file('myfile.idx')
f_read = open('myfile.idx', 'rb')

read_array = idx2numpy.convert_from_file(f_read)
string = f_read.read()
read_array = idx2numpy.convert_from_string(string)


idx2numpy.convert_to_file('myfile_copy.idx', read_array)
f_write = open('myfile_copy2.idx', 'wb')
idx2numpy.convert_to_file(f_write, read_array)
string = idx2numpy.convert_to_string(read_array)


class Image:
    #
    self = np.array()

