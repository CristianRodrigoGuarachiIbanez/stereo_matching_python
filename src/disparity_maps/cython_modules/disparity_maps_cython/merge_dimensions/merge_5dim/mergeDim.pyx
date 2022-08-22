
# distutils: language = c++
from libc.stdlib cimport malloc, free
from numpy cimport ndarray, int64_t, uint8_t, float32_t
from numpy import asarray, int64, uint8, transpose, float32
from cython cimport boundscheck, wraparound
ctypedef unsigned char uchar

cdef class MergeArrayDimensions:
    cdef uchar ****array
    cdef:
        int dim,dim1, dim2, dim12,dim3,dim4,dim5
    def __cinit__(self, uchar[:,:,:,:,:]img):

        self.dim = img.shape[0] #None
        self.dim1 = img.shape[1]#7
        self.dim2 = img.shape[2] #8

        self.dim12 = self.dim1 * self.dim2# 7*8 =56

        self.dim3 = img.shape[3]#120
        self.dim4 = img.shape[4] # 160

        self.array = <unsigned char****>malloc(self.dim*sizeof(unsigned char***))
        self.allocate_memory()
        if not(self.array):
            raise MemoryError()
        self.fillUpMemory()
        self.mergeDimensions(img)
    def __deallocate__(self):
        free(self.array)
    @wraparound(False)
    @boundscheck(False)
    cdef void allocate_memory(self) nogil:
        cdef:
            unsigned int i, ii, j, k, n
        for i in range(self.dim):
            self.array[i] = <uchar***>malloc(self.dim12 *sizeof(uchar**))
            for ii in range(self.dim12):
                self.array[i][ii] = <uchar**>malloc(self.dim3 *sizeof(uchar*))
                for j in range(self.dim3):
                    self.array[i][ii][j] = <uchar*>malloc(self.dim4*sizeof(uchar))

    @wraparound(False)
    @boundscheck(False)
    cdef void fillUpMemory(self) nogil:
        cdef:
            int i,j,k,n
        for i in range(self.dim):
            for j in range(self.dim12):
                for k in range(self.dim3):
                    for n in range(self.dim4):
                        self.array[i][j][k][n]=0
    @wraparound(False)
    @boundscheck(False)
    cdef void mergeDimensions(self, uchar[:,:,:,:,:] img ) nogil:
        cdef:
            int i, j,k, n, m, p
            int jj=0
        for i in range(self.dim):
            for j in range(self.dim1):
                for k in range(self.dim2):
                    for n in range(self.dim3):
                        for m in range(self.dim4):
                            self.array[i][jj][n][m] = img[i,j,k,n,m]
                    # print(k, n, jj)
                    if(jj==self.dim12-1):
                        jj=0
                    else:
                        jj+=1


    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[uint8_t, ndim=4] red_dim_numpy(self):
        cdef int i, j, k, n, m

        output = []
        #print("size nulls", self.s)
        for i in range(self.dim):
            inOutput=[]
            for m in range(self.dim12):
                inOutput0 = []
                for j in range(self.dim3):
                    inOutput1 = []
                    for k in range(self.dim4):
                        inOutput1.append(self.array[i][m][j][k])
                    inOutput0.append(inOutput1)
                inOutput.append(inOutput0)
            output.append(inOutput)

        #print(output)
        return asarray(output, dtype=uint8)

    def get_array(self):
        return self.red_dim_numpy()