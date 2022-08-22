# distutils: language = c++
from libcpp.vector cimport vector
from cython.parallel import parallel, prange
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython cimport boundscheck, wraparound, view
ctypedef unsigned char uchar

cdef struct FREQUENCE:
    uchar key
    int value
ctypedef FREQUENCE freq

cdef class SumImageAnalysisManager:
    cdef:
        vector[freq] frequency
        vector[vector[freq]] frequencies
        int maxValue, minValue
        freq * MAX
        freq *MIN
        int dimN,dimU,dimD
    def __cinit__(self, uchar[:,:,:] image):
        self.dimN = image.shape[0] #7
        self.dimU = image.shape[1] #120
        self.dimD = image.shape[2] #160

        self.calculateFrequencies(image)
        self.MAX = <FREQUENCE*>PyMem_Malloc(self.frequencies.size()*sizeof(FREQUENCE))
        self.MIN = <FREQUENCE*>PyMem_Malloc(self.frequencies.size()*sizeof(FREQUENCE))
        if not(self.MAX or self.MIN):
            raise MemoryError()
        self.extremValues()

    @boundscheck(False)
    @wraparound(False)
    cdef void extremValues(self) nogil:
        cdef:
            int i,j
            int length = self.frequencies.size()
        for i in range(length):
            self.calculateMaxValue(self.frequencies[i])
            self.MAX[i].key = i
            self.MAX[i].value = self.maxValue
            self.calculateMinValue(self.frequencies[i])
            self.MIN[i].key = i
            self.MIN[i].value = self.minValue

    @boundscheck(False)
    @wraparound(False)
    cdef void calculateMaxValue(self, vector[freq] frequency) nogil:
        cdef:
            int j,value
        self.maxValue=0
        value = 0
        for j in range(frequency.size()):
            if(frequency[j].value>value):
                value = frequency[j].value
                self.maxValue = frequency[j].key
    @boundscheck(False)
    @wraparound(False)
    cdef void calculateMinValue(self, vector[freq] frequency) nogil:
        cdef:
            int j, value
        self.minValue=0
        value = 5
        for j in range(frequency.size()):
            if(frequency[j].value<value ):
                value = frequency[j].value
                self.minValue = frequency[j].key
    @boundscheck(False)
    @wraparound(False)
    cdef void calculateFrequencies(self, uchar[:,:,:] image) nogil:
        cdef:
            unsigned int i, j, k
            freq curr

        for i in range(self.dimN):
            for j in range(self.dimU):
                for k in range(self.dimD):
                    curr.key = image[i,j,k]
                    curr.value = 1
                    if not (self.findKey(self.frequency, curr.key)):
                        self.frequency.push_back(curr)

                    else:
                        self.addValues(curr)

                    if(j+1==self.dimU and k+1==self.dimD):
                        self.frequencies.push_back(self.frequency)
                        self.frequency.clear()
    @boundscheck(False)
    @wraparound(False)
    cdef bint findKey(self, vector[freq] f, int key) nogil:
        cdef int i
        for i in range(f.size()):
            if(f[i].key == key):
                return 1 # the key is already in Vector
            else:
                #print("key is not in Vector Index nr.:" + str(i))
                pass
        return 0
    @boundscheck(False)
    @wraparound(False)
    cdef void addValues(self, freq currItem) nogil:
        cdef int i
        for i in range(self.frequency.size()):
            if(self.frequency[i].key == currItem.key):
                self.frequency[i].value += currItem.value
                #print(self.frequency[i].value)
    @boundscheck(False)
    @wraparound(False)
    cdef dict convertToDict(self):
        cdef int i
        frequencies = {}
        for i in range(self.frequencies.size()):
            frequency ={}
            for j in range(self.frequencies[i].size()):
                frequency[self.frequencies[i][j].key] = self.frequencies[i][j].value
            frequencies[i] = frequency
        return frequencies

    cdef dict convertExtremValuesToDict(self, freq* s):
        cdef:
            unsigned int i
            int length = self.frequencies.size()
        #print("LEN: ",length)
        items ={}
        for i in range(length):
            items[s[i].key] =s[i].value
        return items

    def get_frequencies(self):
        return self.convertToDict()
    def get_max_value(self):
        return self.convertExtremValuesToDict(self.MAX)
    def get_min_value(self):
        return self.convertExtremValuesToDict(self.MIN)