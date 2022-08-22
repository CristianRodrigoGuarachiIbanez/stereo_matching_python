

# distutils: language = c++
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from numpy cimport ndarray, float32_t
from numpy import concatenate, asarray, zeros, float32, transpose
from cython cimport boundscheck, wraparound, view


cdef class MergeDisparityArrayDim:
    cdef:
        vector[float***] imgs
        float ***img
        int dim,dim2, dim3
    def __cinit__(self, float[:,:,:,:,:]img):

        self.dim = img.shape[1]*img.shape[2] #56
        self.dim2 = img.shape[3] #120
        self.dim3 = img.shape[4] #160
        self.img = <float***>malloc(self.dim*sizeof(float**))
        self.allocate_memory()
        self.populate()
        if not(self.img):
            raise MemoryError()
        self.merge_dimensions(img)

    cdef void allocate_memory(self) nogil:
        cdef int i, j
        for i in range(self.dim):
            self.img[i] = <float**>malloc(self.dim2*sizeof(float*))
            for j in range(self.dim2):
                self.img[i][j] = <float*>malloc(self.dim3*sizeof(float))

    cdef void populate(self) nogil:
        cdef int i,j,k
        for i in range(self.dim):
            for j in range(self.dim2):
                for k in range(self.dim3):
                    self.img[i][j][k]=0.0

    @wraparound(False)
    @boundscheck(False)
    cdef void mergeDimensions(self, float[:,:,:,:] image ) nogil:
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
    cdef void save_array(self, float***img) nogil:
        self.imgs.push_back(img)

    @wraparound(False)
    @boundscheck(False)
    cdef void merge_dimensions(self, float[:,:,:,:,:] img) nogil:
        cdef int i
        for i in range(img.shape[0]):
            self.mergeDimensions(img[i,:,:,:,:])
            self.save_array(self.img)

    @wraparound(False)
    @boundscheck(False)
    @staticmethod
    cdef MergeDisparityArrayDim c_wrapper(float[:,:,:,:,:] img):
        """Factory function to create WrapperClass objects from
        given data type"""

        cdef MergeDisparityArrayDim wrapper = MergeDisparityArrayDim.__new__(MergeDisparityArrayDim, img)
        #wrapper.merge_dimensions(img)
        #print(wrapper.imgs.size())
        return wrapper

    @wraparound(False)
    @boundscheck(False)
    cdef ndarray[float32_t, ndim=4] vectorToList(self, float[:,:,:,:,:] img=None ):
        cdef:
            int i, j, k, n, m
            float ***array
            vector[float***] imgs
        if(img is None):
            imgs = self.imgs
            print("VECTOR SIZE",imgs.size())
        else:
            im = MergeDisparityArrayDim.c_wrapper(img)
            imgs = im.imgs
            print("Vector size", imgs.size())

        output = []
        #print("size nulls", self.s)
        for i in range(self.imgs.size()):
            array = self.imgs[i]
            inOutput=[]
            for m in range(self.dim):
                inOutput0 = []
                for j in range(self.dim2):
                    inOutput1 = []
                    for k in range(self.dim2):
                        inOutput1.append(array[m][j][k])
                    inOutput0.append(inOutput1)
                inOutput.append(inOutput0)
            output.append(inOutput)

        #print(output)
        return asarray(output, dtype=float32)

    def get_vector(self, img):
        return self.vectorToList(img)