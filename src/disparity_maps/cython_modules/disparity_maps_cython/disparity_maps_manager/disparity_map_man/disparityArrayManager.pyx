
from numpy cimport ndarray, uint8_t
from numpy import concatenate, asarray, zeros, uint8, transpose
from cython cimport boundscheck, wraparound, view
from libc.math cimport ceil
ctypedef unsigned char uchar

cdef class DisparityArrayManager:
    cdef:
        uchar[:,:,:,:] img56
        uchar[:,:,:]img
        int img_count
        int s_img_count
        int DO,DT,DTH,DF
    def __cinit__(self, uchar[:,:,:,:] image, int h_plot, int v_plot):
        self.DO = image.shape[0] #160
        self.DT = image.shape[1] #120
        self.DTH = image.shape[2] #7
        self.DF = image.shape[3] #8

        self.img = zeros((self.DTH, self.DT, self.DO), dtype=uint8)
        self.img56 = zeros((self.DTH,self.DF,self.DT, self.DO), dtype=uint8)

        self.s_img_count = h_plot * v_plot
        self.img_count = image.shape[2] * image.shape[3]

        self.recover_imgs(image)

    @boundscheck(False)
    @wraparound(False)
    cdef void recover_imgs(self, uchar[:,:,:,:] plot_image):
        cdef:
            int i,j,n,z
            uchar[:,:] imageT
        cdef:
            int h_step = 0
            int v_step = 0
            uchar[:,:] image = zeros((120,160),dtype=uint8)
            uchar[:,:,:] images = zeros((self.DF, self.DT, self.DO), dtype=uint8)
            double c = ceil(self.img_count / self.s_img_count)
            int length = <int>c
        if (length ==0):
           length = 1;

        #print(self.s_img_count) #64
        #print(length)
        for i in range(length):
            print(i)
            for j in range(self.s_img_count):
                if (j + i * self.s_img_count) < self.img_count:
                    imageT = transpose(plot_image[:, :, h_step, v_step], axes=(1,0))
                    #print("image shape",h_step, v_step, asarray(imageT).shape)
                    if(imageT.ndim>0):
                        image = image + asarray(imageT, dtype=uint8)
                        #print("image", image.shape)
                        for z in range(self.DT):
                            for n in range(self.DO):
                                images[v_step,z,n] = imageT[z,n]
                else:
                    break
                v_step += 1
                if(v_step == plot_image.shape[3]):
                    self.recover_56imgs(images, h_step)
                    self.recover_summed_imgs(image, h_step)
                    image = zeros((120,160),dtype=uint8)
                    v_step = 0
                    h_step += 1
        #print(asarray(image).shape)
    @boundscheck(False)
    @wraparound(False)
    cdef void recover_summed_imgs(self, uchar[:,:] imgs, int index):
        cdef:
            int k, m
        for k in range(self.DT):
            for m in range(self.DO):
                    self.img[index,k,m] = imgs[k,m]
    @boundscheck(False)
    @wraparound(False)
    cdef void recover_56imgs(self, uchar[:,:,:] imgs, int index):
        cdef:
            int j, k, m
        for j in range(self.DF):
            for k in range(self.DT):
                for m in range(self.DO):
                        self.img56[index,j,k,m] = imgs[j,k,m]
    @boundscheck(False)
    @wraparound(False)
    cdef ndarray[uchar, ndim=3] getImgs(self):
        return asarray(self.img, dtype=uint8)
    @boundscheck(False)
    @wraparound(False)
    cdef ndarray[uchar, ndim=4] get56Imgs(self):
        return asarray(self.img56, dtype=uint8)
    def getImage(self):
        return self.getImgs()
    def get56Image(self):
        return self.get56Imgs()