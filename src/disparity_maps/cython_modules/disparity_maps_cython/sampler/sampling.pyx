from numpy cimport ndarray, uint8_t
from numpy import concatenate, asarray, zeros, uint8, transpose
from cython cimport boundscheck, wraparound, view
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport ceil
ctypedef unsigned char uchar
cdef class CSampler:
    cdef:
        uchar ***** samples
        int dim2,dim3,dim4, dim5, amount
    def __cinit__(self, uchar[:,:,:,:,:] images, int amount):
        self.amount = amount
        self.dim2 = images.shape[1]
        self.dim3 = images.shape[2]
        self.dim4 = images.shape[3]
        self.dim5 = images.shape[4]

        self.samples=<unsigned char*****>PyMem_Malloc(self.amount*sizeof(unsigned char****))
        self.allocate_memory()
        self.populate()
        self.extract_samples(images)
        if not (self.samples):
             raise MemoryError()
    def __deallocate__(self):
        PyMem_Free(self.samples)

    @wraparound(False)
    @boundscheck(False)
    cdef void allocate_memory(self):
        cdef unsigned int i,j,k,n
        for i in range(self.amount):
            self.samples[i] = <unsigned char****>PyMem_Malloc(self.dim2*sizeof(unsigned char***))
            for j in range(self.dim2):
                self.samples[i][j]=<uchar***>PyMem_Malloc(self.dim3*sizeof(uchar**))
                for k in range(self.dim3):
                    self.samples[i][j][k]=<uchar**>PyMem_Malloc(self.dim4*sizeof(uchar*))
                    for n in range(self.dim4):
                        self.samples[i][j][k][n]=<uchar*>PyMem_Malloc(self.dim5*sizeof(uchar))

    @wraparound(False)
    @boundscheck(False)
    cdef void populate(self) nogil:
        cdef unsigned int i, j, k, n, m
        for i in range(self.amount):
            for j in range(self.dim2):
                for k in range(self.dim3):
                    for n in range(self.dim4):
                        for m in range(self.dim5):
                            self.samples[i][j][k][n][m]=0

    @wraparound(False)
    @boundscheck(False)
    cdef void extract_samples(self, uchar[:,:,:,:,:] images):

        cdef unsigned int i, j, k, n, m
        for i in range(self.amount):
            print("index", i)
            for j in range(self.dim2):
                for k in range(self.dim3):
                    for n in range(self.dim4):
                        for m in range(self.dim5):
                            #print(images[i,j,k,n,m])
                            self.samples[i][j][k][n][m]=images[i,j,k,n,m]

    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[uint8_t, ndim=5] convert_to_python(self, uchar*****images):

        cdef unsigned int i, j, k, n, m
        m1 =[]
        for i in range(self.amount):
            m2=[]
            for j in range(self.dim2):
                m3 = []
                for k in range(self.dim3):
                    m4 =[]
                    for n in range(self.dim4):
                        m5 =[]
                        for m in range(self.dim5):
                            m5.append(images[i][j][k][n][m])
                        m4.append(m5)
                    m3.append(m4)
                m2.append(m3)
            m1.append(m2)
        return asarray(m1, dtype=uint8)
    def get_img(self):
        return self.convert_to_python(self.samples)