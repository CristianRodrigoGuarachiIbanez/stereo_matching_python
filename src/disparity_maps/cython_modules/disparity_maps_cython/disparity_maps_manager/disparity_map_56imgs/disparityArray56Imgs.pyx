from numpy cimport ndarray, uint8_t
from numpy import concatenate, asarray, zeros, uint8, transpose
from cython cimport boundscheck, wraparound
ctypedef unsigned char uchar

cdef class DisparityArray56Manager:
    cdef:
        uchar[:,:,:] img56
        int img_count
        int s_img_count
        int DO,DT,DTH,DF
    def __cinit__(self, uchar[:,:,:,:] image, int h_plot, int v_plot):
        self.DO = image.shape[0] #160
        self.DT = image.shape[1] #120
        self.DTH = image.shape[2] #7
        self.DF = image.shape[3] #8

        self.img56 = zeros((self.DTH*self.DF,self.DT, self.DO), dtype=uint8)

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
            uchar[:,:,:] images = zeros((self.img_count, self.DT, self.DO), dtype=uint8)
        i =0
        for j in range(self.img_count):
            if (j + i * self.s_img_count) < self.img_count:
                imageT = transpose(plot_image[:, :, h_step, v_step], axes=(1,0))
                #print("image shape",h_step, v_step, asarray(imageT).shape)
                if(imageT.ndim>0):
                    for z in range(self.DT):
                        for n in range(self.DO):
                            self.img56[j,z,n] = imageT[z,n]
            else:
                break
            v_step += 1
            if(v_step == plot_image.shape[3]):  #8
                #self.recover_56imgs(images, h_step)
                v_step = 0
                h_step += 1


    @boundscheck(False)
    @wraparound(False)
    cdef ndarray[uchar, ndim=3] get56imgs(self):
        return asarray(self.img56, dtype=uint8)

    def get_56images(self):
        return self.get56imgs()