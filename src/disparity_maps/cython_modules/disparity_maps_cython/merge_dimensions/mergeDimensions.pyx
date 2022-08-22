# distutils: language = c++
from libc.stdlib cimport malloc, free
from numpy cimport ndarray, int64_t, uint8_t, float32_t
from numpy import asarray, int64, uint8, transpose, float32
from cython cimport boundscheck, wraparound
ctypedef unsigned char uchar

cdef class MergeDimensions:
    cdef float ***** array
    cdef:
        int dim,dim1, dim2, dim23,dim3,dim4,dim5
    def __cinit__(self, float[:,:,:,:,:,:]img):

        self.dim = img.shape[0] #None
        self.dim1 = img.shape[1]#10

        self.dim2 = img.shape[2] #7
        self.dim3 = img.shape[3]#8

        self.dim23 = self.dim2 * self.dim3# 7*8 =56
        self.dim4 = img.shape[4] # 120
        self.dim5 = img.shape[5] #160
        self.array = <float*****>malloc(self.dim*sizeof(float****))
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
            self.array[i] = <float****>malloc(self.dim1 *sizeof(float***))
            for ii in range(self.dim1):
                self.array[i][ii] = <float***>malloc(self.dim23 *sizeof(float**))
                for j in range(self.dim23):
                    self.array[i][ii][j] = <float**>malloc(self.dim4*sizeof(float*))
                    for k in range(self.dim4):
                        self.array[i][ii][j][k] = <float*>malloc(self.dim5*sizeof(float))
    @wraparound(False)
    @boundscheck(False)
    cdef void fillUpMemory(self) nogil:
        cdef:
            int i,j,ii,k,n
        for i in range(self.dim):
            for j in range(self.dim1):
                for ii in range(self.dim23):
                    for k in range(self.dim3):
                        for n in range(self.dim4):

                            self.array[i][j][ii][k][n]=0.0
    @wraparound(False)
    @boundscheck(False)
    cdef void mergeDimensions(self, float[:,:,:,:,:,:] img ) nogil:
        cdef:
            int i, j,k, n, m, p
            int jj=0
        for i in range(self.dim):
            for j in range(self.dim1):
                for k in range(self.dim2):
                    for n in range(self.dim3):
                        for m in range(self.dim4):
                            for p in range(self.dim5):
                                self.array[i][j][jj][m][p] = img[i,j,k,n,m,p]
                        # print(k, n, jj)
                        if(jj==self.dim23-1):
                            jj=0
                        else:
                            jj+=1

    @wraparound(False)
    @boundscheck(False)
    @staticmethod
    cdef MergeDimensions c_wrapper(float[:,:,:,:,:,:] img, float *****array, int dim, int dim1, int dim2, int dim3, int dim4, int dim5 ,int dim23):
        """Factory function to create WrapperClass objects from
        given data type"""
        # Call to __new__ bypasses __init__ constructor
        cdef MergeDimensions wrapper = MergeDimensions.__new__(MergeDimensions, img)
        wrapper.dim = dim
        wrapper.dim1 = dim1
        wrapper.dim2 = dim2
        wrapper.dim3 = dim3
        wrapper.dim4 = dim4
        wrapper.dim5 = dim5
        wrapper.dim23 = dim23
        wrapper.array = array
        wrapper.allocate_memory()
        if not(wrapper.array):
            raise MemoryError()
        wrapper.fillUpMemory()

        return wrapper

    @wraparound(False)
    @boundscheck(False)
    @staticmethod
    cdef MergeDimensions redDim(float[:,:,:,:,:,:] img):
        """Factory function to create WrapperClass objects with
        newly allocated my_c_struct"""
        cdef:

            int dim, dim1, dim2, dim3, dim4, dim5, dim23
            float ***** array

        dim= img.shape[0] #None
        dim1 = img.shape[1]#10

        dim2 = img.shape[2] #7
        dim3 = img.shape[3]#8

        dim23 = dim2 * dim3# 7*8 =56

        dim4 = img.shape[4] # 120
        dim5 = img.shape[5] #160
        array = <float*****>malloc(dim*sizeof(float****))
        return MergeDimensions.c_wrapper(img, array, dim, dim1, dim2, dim3, dim4, dim5, dim23)

    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[uint8_t, ndim=5] red_dim_numpy(self, float[:,:,:,:,:,:] img=None ):
        cdef int i, j, k, n, m
        if(img is not None):
            MergeDimensions.redDim(img)
        else:
            pass

        output = []
        #print("size nulls", self.s)
        for i in range(self.dim):
            inOutput=[]
            for m in range(self.dim1):
                inOutput0 = []
                for j in range(self.dim23):
                    inOutput1 = []
                    for k in range(self.dim4):
                        inOutput2 = []
                        for n in range(self.dim5):
                            inOutput2.append(self.array[i][m][j][k][n])
                        inOutput1.append(inOutput2)
                    inOutput0.append(inOutput1)
                inOutput.append(inOutput0)
            output.append(inOutput)

        #print(output)
        return asarray(output, dtype=float32)

    def reshape(self,array):
        return self.red_dim_numpy(array)
    def get_array(self):
        return self.red_dim_numpy(None)