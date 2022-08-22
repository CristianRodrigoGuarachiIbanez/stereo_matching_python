# distutils: language = c++
from libc.stdlib cimport malloc, free
from numpy cimport ndarray, float32_t
from numpy import concatenate, asarray, zeros, float32, transpose
from cython cimport boundscheck, wraparound, view
from libc.math cimport ceil
ctypedef float32_t FLOAT32


cdef class SumDisparityArrays:
    cdef:
        float****array
        int dim, dim1, dim2, dim3, dim4
    def __cinit__(self, float[:,:,:,:,:]img):

        self.dim = img.shape[0] #None

        self.dim1 = img.shape[1]#7
        self.dim2 = img.shape[2] #8

        self.dim3 = img.shape[3]#120
        self.dim4 = img.shape[4] # 160

        self.array = <float****>malloc(self.dim*sizeof(float***))
        self.allocate_memory()
        self.sum_imgs(img)

    def __deallocate__(self):
        free(self.array)

    @wraparound(False)
    @boundscheck(False)
    cdef void allocate_memory(self) nogil:
        cdef:
            unsigned int i, j, k, n
        for i in range(self.dim):
            self.array[i] = <float***>malloc(self.dim1 *sizeof(float**))
            for j in range(self.dim1):
                self.array[i][j] = <float**>malloc(self.dim3 *sizeof(float*))
                for k in range(self.dim3):
                    self.array[i][j][k] = <float*>malloc(self.dim4*sizeof(float))

    @wraparound(False)
    @boundscheck(False)
    cdef void fillUpMemory(self) nogil:
        cdef:
            int i,j,k,n
        for i in range(self.dim):
            for j in range(self.dim1):
                for n in range(self.dim3):
                    for k in range(self.dim4):
                        self.array[i][j][k][n]=0.0
    @boundscheck(False)
    @wraparound(False)
    cdef void sum_imgs(self, float[:,:,:,:,:] images):
        cdef:
            int i,j,n
            float[:,:] image = zeros((120,160),dtype=float32)
        for i in range(self.dim):
            for j in range(self.dim1):
                for n in range(self.dim2):
                    image = image + asarray(images[i,j,n], dtype=float32)
                    #print("index:", i, "j:", j, "n:", n, "dim:", image.shape)
                    if(n == images.shape[2]-1):
                        self.recover_summed_imgs(image, i,j)
                        image = zeros((120,160),dtype=float32)

    @boundscheck(False)
    @wraparound(False)
    cdef void recover_summed_imgs(self, float[:,:] imgs, int index1, int index2):
        cdef:
            int k, m, n
        for k in range(self.dim3):
            for m in range(self.dim4):
                    self.array[index1][index2][k][m] = imgs[k,m]

    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[float32_t, ndim=4] getImgArrays(self ):
        cdef int i, j, k, n, m

        output = []
        #print("size nulls", self.s)
        for i in range(self.dim):
            inOutput=[]
            for m in range(self.dim1):
                inOutput0 = []
                for j in range(self.dim3):
                    inOutput1 = []
                    for k in range(self.dim4):
                        inOutput1.append(self.array[i][m][j][k])
                    inOutput0.append(inOutput1)
                inOutput.append(inOutput0)
            output.append(inOutput)

        #print(output)
        return asarray(output, dtype=float32)

    def get_array(self):
        return self.getImgArrays()
