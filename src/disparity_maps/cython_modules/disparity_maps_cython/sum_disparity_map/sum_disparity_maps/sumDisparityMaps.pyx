
from numpy cimport ndarray, uint8_t
from numpy import concatenate, asarray, zeros, uint8, transpose
from cython cimport boundscheck, wraparound, view
from libc.math cimport ceil
ctypedef unsigned char uchar

cdef class SumDisparityArray:
    cdef:
        uchar[:,:,:]img
        int DO,DT,DTH,DF
    def __cinit__(self, uchar[:,:,:,:] image):
        self.DO = image.shape[0] #7
        self.DT = image.shape[1] #8
        self.DTH = image.shape[2] #120
        self.DF = image.shape[3] #160

        self.img = zeros((self.DO, self.DTH, self.DF), dtype=uint8)

        self.sum_imgs(image)

    @boundscheck(False)
    @wraparound(False)
    cdef void sum_imgs(self, uchar[:,:,:,:] images):
        cdef:
            int i,j,n,z
            uchar[:,:] image = zeros((120,160),dtype=uint8)

        for i in range(self.DO):
            for j in range(self.DT):
                image = image + asarray(images[i,j], dtype=uint8)
                #print("index", j, images.shape[1])
                if(j == images.shape[1]-1):
                    self.recover_summed_imgs(image, i)
                    image = zeros((120,160),dtype=uint8)

    @boundscheck(False)
    @wraparound(False)
    cdef void recover_summed_imgs(self, uchar[:,:] imgs, int index):
        cdef:
            int k, m
        for k in range(self.DTH):
            for m in range(self.DF):
                    self.img[index,k,m] = imgs[k,m]
    @boundscheck(False)
    @wraparound(False)
    cdef ndarray[uchar, ndim=3] getImgs(self):
        return asarray(self.img, dtype=uint8)
    def getImage(self):
        return self.getImgs()
