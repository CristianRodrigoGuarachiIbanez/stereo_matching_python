from disparity_maps.iCub_Vergence.Vergence_Control_cpp import VergenceControl as VC_Cpp
from disparity_maps.iCub_Vergence.parameters import params
from disparity_maps.plot_image.plot_4DArray import show_4Darray_matplot, show_3Darray_plot, _show_4Darray_plot
from disparity_maps.cython_modules.disparity_maps_cython.disparity_maps_manager.disparity_map_sums.disparityArraySum import DisparityArraySum
from disparity_maps.cython_modules.disparity_maps_cython.disparity_maps_manager.disparity_map_56imgs.disparityArray56Imgs import DisparityArray56Manager
from disparity_maps.dataset_tools.h5Writer import  H5Writer
from cv2 import imshow, waitKey, imread
from PIL import Image
from pickle import load
from h5py import File
from os import chdir, makedirs
from os.path import exists, abspath
from numpy import int32, uint8, save, asarray, float32



from typing import List
class DisparityMaps:

     def __init__(self):
         self.vc_cpp = VC_Cpp.VergenceControl(params['resVisualbin'][0], params['resVisualbin'][1], abspath("./iCub_Vergence/data/Gt43B0.0208f0.063ph7.ini"))
         self.disparity_array = list()
         self.disparity_array56 = list()

     def get_disparity_arrays(self):
         return asarray(self.disparity_array,dtype=uint8)

     def get_disparity_56arrays(self):
         return asarray(self.disparity_array56, dtype=uint8)

     def load_imgs(self, right_img, left_img):
         print(left_img.shape, "vs", right_img.shape)
         self.vc_cpp.loadImgArr(right_img, "r")
         self.vc_cpp.loadImgArr(left_img, "l")

     def construct_disparity_maps(self):
         self.vc_cpp.computeFilterPatch()

     def calculate_disparity_maps(self, right_img, left_img, sum=False):
         self.length: int = right_img.shape[0]
         for i in range(self.length):
             self.load_imgs(asarray(left_img[i], dtype=float32), asarray(right_img[i],dtype=float32))
             self.construct_disparity_maps()
             disparity_map = self.vc_cpp.getV1compResponse_1_2()
             self.format_disparity_arrays(disparity_map=asarray(disparity_map,dtype=uint8), sum=sum)
             print("index:", i, "length:",self.length, "len sum of arrays:", len(self.disparity_array56))

     def format_disparity_arrays(self, disparity_map, sum=False)->None:
         if(sum is False):
             dam = DisparityArray56Manager(disparity_map, 8, 8)
             img = dam.get_56images()
             self.disparity_array56.append(img)
         else:
            dam = DisparityArraySum(disparity_map, 8,8)
            img3D = dam.get_images()
            self.disparity_array.append(img3D)

     def show_img(self, array)->None:
         print("SHAPE:", array.shape)
         if(len(array.shape) ==3):
             show_3Darray_plot(array, "Disparity Maps Sum")

         elif(len(array.shape)==4):
             _show_4Darray_plot(array, "Disparity Maps")

     def plot_4DImage(self, image)->None:
         show_4Darray_matplot(image, 8,8)

     def save_sum_disparity_maps_png(self, path_disparity="/scratch/gucr/tEDRAM2/training_data/disparity_maps"):
         if not exists(path_disparity):
             makedirs(path_disparity)
         chdir(path_disparity)
         assert (len(self.disparity_array)>0), "there is no image"
         for i in range(self.length):
             for j in range(self.get_disparity_arrays().shape[1]):
                 img = Image.fromarray(self.get_disparity_arrays()[i, j, :, :])
                 img.save(path_disparity + "/disparity_map_" + str(i) + str(j) + ".png")

     def save_disparity_maps_as_png(self, path_disparity="/scratch/gucr/tEDRAM2/training_data/disparity_maps"):
         if not exists(path_disparity):
             makedirs(path_disparity)
         chdir(path_disparity)
         assert (len(self.disparity_array56)>0), "there is no image"
         rows= 0
         cols = 0
         for i in range(self.length):
             if (cols > 10):
                 cols = 0
             for j in range(self.get_disparity_56arrays().shape[1]):
                 #show_4Darray_matplot(self.get_disparity_56arrays()[i,:,:,:], "disparity map", 8,8, i,)
                 #imshow("disparity", self.get_disparity_56arrays()[i,j,:,:])
                 img = Image.fromarray(self.get_disparity_56arrays()[i,j,:,:])
                 img.save(path_disparity + "/disparity_map_"+str(rows)+str(cols)+"_"+str(j)+".png")
                 if(j==55):
                     cols+=1
             if (i == 10 or i ==21 or i==32 or i==43 or i==54):
                 rows += 1

     def save_disparity_maps_h5(self, path:str ="./training_data/disparity_maps_56imgs.h5"):
         writer: H5Writer = H5Writer(path)
         datasets:List[str] = [ "disparity_arrays_56" ] # "disparity_arrays_s"
         disparity_map_data = [asarray(self.disparity_array56, dtype=uint8)] # asarray(self.disparity_array, dtype=uint8)
         writer.saveImgDataIntoGroup(disparity_map_data, "disparity_arrays", datasetNames=datasets)

     def open_file(self, filename_l:str, filename_r:str)->None:
         imgL = imread(filename_l)
         imgR = imread(filename_r)
         factor = 90
         new_shape = (int(239*factor/100), int(137*factor/100))
         print("NEW SHAPE:",new_shape)
         right_img = resize(imgL, new_shape)
         left_img = resize(imgR, new_shape)
         self.load_imgs(asarray(left_img[:,:,0], dtype=float32), asarray(right_img[:,:,0], dtype=float32))
         self.construct_disparity_maps()
         disparity_map = self.vc_cpp.getV1compResponse_1_2()
         print("4D Image:", disparity_map.shape)
         show_4Darray_matplot(disparity_map,"Disparity Maps", 8,8)
         self.format_disparity_arrays(disparity_map=asarray(disparity_map, dtype=uint8))

class DisparityMapsDisplay:

    def __init__(self, filename_bino=None, filename_l=None)->None:

        self._open_h5(filename=filename_bino)
        self._open_labels(filename=filename_l)

    def _open_h5(self, filename:str)->None:
        arr= File(filename, 'r')
        self.left_img = arr['binocular_image']['left_img'][41800:42000, :, :]
        self.right_img = arr['binocular_image']['right_img'][41800:42000, :, :]

    def _open_disparity_maps(self, filename:str)->None:
        with open(filename, "rb") as file:
            self.img = asarray(load(file), dtype=uint8)
            file.close()

    def _open_labels(self, filename:str):
        with open(filename, "rb") as file:
            self.grTruth = asarray(load(file), dtype=int32)[41800:42000, :]
            file.close()

    def display_disparity_arrays(self)->None:
        dim:int = self.img.shape[0]
        dim1 = self.img.shape[1]
        for i in range(0,dim):
            print("Ground Truth -> ", self.grTruth[i])
            for j in range(dim1):
                print("shape ->", self.img[i, j, :, :].shape)
                imshow("./training_data/disparity_maps/image_" + str(i) + str(j)+".png", self.img[i, j, :, :])
                waitKey(0)

    def display_binocular(self):
        assert(self.left_img.shape[0] == self.right_img.shape[0]), "Eye Images dont have the same size"
        dim: int = self.left_img.shape[0]
        dim1 = self.left_img.shape[1]
        for i in range(0, dim):
            print("Ground Truth -> ", self.grTruth[i])

            imshow("left", self.left_img[i, :, :])
            imshow("right", self.right_img[i, :, :])
            waitKey(0)

if __name__ == '__main__':
    pass
