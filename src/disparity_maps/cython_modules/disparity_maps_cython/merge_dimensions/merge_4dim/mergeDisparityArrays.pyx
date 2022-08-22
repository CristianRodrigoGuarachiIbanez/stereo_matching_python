from libc.stdlib cimport malloc, free
from numpy cimport ndarray, uint8_t
from numpy import concatenate, asarray, uint8
from cython cimport boundscheck, wraparound

ctypedef unsigned char uchar


cdef class MergeDisparityArrays:
    cdef:

        uchar ***img
        int dim,dim2, dim3
    def __cinit__(self, uchar[:,:,:,:] img):


        self.dim = img.shape[0]*img.shape[1] #56
        self.dim2 = img.shape[2] #120
        self.dim3 = img.shape[3] #160
        self.populate()
        if not(self.img):
            raise MemoryError()
        self.mergeDimensions(img)

    @wraparound(False)
    @boundscheck(False)
    cdef void allocate_memory(self) nogil:
        self.img = <uchar***>malloc(self.dim*sizeof(uchar**))
        cdef int i, j
        for i in range(self.dim):
            self.img[i] = <uchar**>malloc(self.dim2*sizeof(uchar*))
            for j in range(self.dim2):
                self.img[i][j] = <uchar*>malloc(self.dim3*sizeof(uchar))
    @wraparound(False)
    @boundscheck(False)
    cdef void populate(self) nogil:
        self.allocate_memory()
        cdef int i,j,k
        for i in range(self.dim):
            for j in range(self.dim2):
                for k in range(self.dim3):
                    self.img[i][j][k]=0

    @wraparound(False)
    @boundscheck(False)
    cdef void mergeDimensions(self, uchar[:,:,:,:] image ) nogil:
        cdef:
            int i,j,k, n
            int jj=0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(self.dim2):
                    for n in range(self.dim3):
                        self.img[jj][k][n] = image[i,j,k,n]
                #print(i,j,jj)
                if(jj==self.dim-1):
                    jj=0
                else:
                    jj+=1


    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[uint8_t, ndim=3] vectorToList(self ):

        cdef:
            int i, j, k, n, m
        output = []
        #print("Vector size", img.size())
        for m in range(self.dim):
            inOutput0 = []
            for j in range(self.dim2):
                inOutput1 = []
                for k in range(self.dim3):
                    inOutput1.append(self.img[m][j][k])
                inOutput0.append(inOutput1)
            output.append(inOutput0)

        #print(output)
        return asarray(output, dtype=uint8)

    def get_vector(self):
        return self.vectorToList()