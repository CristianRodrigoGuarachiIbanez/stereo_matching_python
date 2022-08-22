# distutils: language = c++
from libc.stdlib cimport malloc, free
from numpy cimport ndarray, int64_t, uint8_t
from numpy import asarray, int64, int8
from cython cimport boundscheck, wraparound
ctypedef unsigned char uchar

cdef class MergeDimensions:
    cdef uchar**** array
    cdef:
        int dim,dim1, dim2, dim12,dim3,dim4
    def __cinit__(self, uchar[:,:,:,:,:]img):

        self.dim = img.shape[0] #10
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
            unsigned int i, j, k, n
        for i in range(self.dim):
            self.array[i] = <uchar***>malloc(self.dim12 *sizeof(uchar**))
            for j in range(self.dim12):
                self.array[i][j] = <uchar**>malloc(self.dim3*sizeof(uchar*))
                for k in range(self.dim3):
                    self.array[i][j][k] = <uchar*>malloc(self.dim4*sizeof(uchar))
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
            int i, j, k, n, m
            int jj=0
        for i in range(self.dim):
            for j in range(self.dim1):
                for k in range(self.dim2):
                    for n in range(self.dim3):
                        for m in range(self.dim4):
                            self.array[i][jj][n][m] = img[i,j,k,n,m]
                    #print(j, k, jj)
                    if(jj==self.dim12-1):
                        jj=0
                    else:
                        jj+=1


    @wraparound(False)
    @boundscheck(False)
    @staticmethod
    cdef MergeDimensions c_wrapper(uchar[:,:,:,:,:] img, uchar ****array, int dim, int dim1, int dim2, int dim3, int dim4, int dim12):
        """Factory function to create WrapperClass objects from
        given data type"""
        # Call to __new__ bypasses __init__ constructor
        cdef MergeDimensions wrapper = MergeDimensions.__new__(MergeDimensions, img)
        wrapper.dim = dim
        wrapper.dim1 = dim1
        wrapper.dim2 = dim2
        wrapper.dim3 = dim3
        wrapper.dim4 = dim4
        wrapper.dim12 = dim12
        wrapper.array = array
        wrapper.allocate_memory()
        if not(wrapper.array):
            raise MemoryError()
        wrapper.fillUpMemory()

        return wrapper

    @wraparound(False)
    @boundscheck(False)
    @staticmethod
    cdef MergeDimensions redDim(uchar[:,:,:,:,:] img):
        """Factory function to create WrapperClass objects with
        newly allocated my_c_struct"""
        cdef:

            int dim, dim1, dim2, dim3, dim4, dim12
            uchar **** array

        dim= img.shape[0] #10
        dim1 = img.shape[1]#7
        dim2 = img.shape[2] #8
        dim12 = dim1 * dim2# 7*8 =56
        dim3 = img.shape[3]#120
        dim4 = img.shape[4] # 160
        array = <unsigned char****>malloc(dim*sizeof(unsigned char***))
        return MergeDimensions.c_wrapper(img, array, dim, dim1, dim2, dim3, dim4, dim12)

    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[int64_t, ndim=1] red_dim_numpy(self, uchar[:,:,:,:,:] img=None ):
        cdef int i, j, k, n, m
        if(img is None):
            MergeDimensions.redDim(img)
        else:
            pass

        output = []
        #print("size nulls", self.s)
        for i in range(self.dim):
            inOutput=[]
            for j in range(self.dim12):
                inOutput1 = []
                for k in range(self.dim3):
                    inOutput2 = []
                    for n in range(self.dim4):
                        inOutput2.append(self.array[i][j][k][n])
                    inOutput1.append(inOutput2)
                inOutput.append(inOutput1)
            output.append(inOutput)

        #print(output)
        return asarray(output, dtype=int8)


    def reshape(self,array):
        return self.red_dim_numpy(array)
    def get_array(self):
        return self.red_dim_numpy(None)