
from disparity_maps.stereovision import DisparityMaps
from os import makedirs, listdir
from os.path import exists, dirname, abspath
from cv2 import imshow, waitKey, imread, destroyAllWindows, resize, imwrite, COLOR_BGR2RGB
from numpy import ndarray, int32, uint8, save, asarray, float32
from PIL import Image
import pytest

path = dirname(abspath(__file__))

def gather_dir(path = r"./training_data/binocular_imgs/bino/"):
    dir = sorted(listdir(path))
    print(dir)
    imgs = []
    for i in range(len(dir)):
        img = imread(path+dir[i],COLOR_BGR2RGB)
        print("image dir ->", img.shape)
        imgs.append(img)
    return asarray(imgs, dtype=float32)

def bi_imsave(arr=None, arr2=None, path_disparity = "/scratch/gucr/tEDRAM2/training_data/binocular_imgs"):

    if not exists(path_disparity):
        makedirs(path_disparity)
    for i in range(arr.shape[0]):
        if(arr is not None):
            img_left = Image.fromarray(arr[i, :, :])
            img_left.save(path_disparity + "/binocular_img_left" + str(i) + ".png")
        if(arr2 is not None):
            img_right=Image.fromarray(arr2[i,:,:])
            img_right.save(path_disparity+ "/binocular_img_right" + str(i) + ".png")


@pytest.mark.parametrize(
    "path_left_image, path_right_image, path_output", [(path + "/binocular_imgs/left/", path + "/binocular_imgs/right/", path + "/binocular_imgs/output/")]
)
def test_disparity_maps(path_left_image, path_right_image, path_output):
    DPM: DisparityMaps = DisparityMaps()
    left_img = gather_dir(path=path_left_image)
    right_img = gather_dir(path=path_right_image)
    print("images -> left side {}  right side {}".format(left_img.shape, right_img.shape))

    DPM.calculate_disparity_maps(right_img=right_img, left_img=left_img, sum=False)
    # disparity_maps_56: ndarray = DPM.get_disparity_56arrays()
    disparity_maps: ndarray = DPM.get_disparity_arrays()
    print("SHAPE Disparity Maps:", disparity_maps.shape)

    # DPM.save_disparity_maps_h5()
    DPM.save_disparity_maps_as_png(path_disparity=path_output)
    # DPM.save_sum_disparity_maps_png(path_disparity="/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/training_data/disparity_maps/")

if (__name__ == "__main__"):
    DPM: DisparityMaps = DisparityMaps()
    left_img = gather_dir(path=path + "/binocular_imgs/left/")
    right_img = gather_dir(path=path + "/binocular_imgs/right/")
    print("images -> left side {}  right side {}".format(left_img.shape, right_img.shape))

    DPM.calculate_disparity_maps(right_img=right_img, left_img=left_img, sum=False)
    # disparity_maps_56: ndarray = DPM.get_disparity_56arrays()
    disparity_maps: ndarray = DPM.get_disparity_arrays()
    print("SHAPE Disparity Maps:", disparity_maps.shape)

    # DPM.save_disparity_maps_h5()
    DPM.save_disparity_maps_as_png(path_disparity=path + "/binocular_imgs/output/")

    """
    path: str = r"./training_data/scene_image_data.h5"
    arr = File(path, 'r')
    img_scene = arr['scene_img'][41400:41600, :, :]
    bi_imsave(img_scene, path_disparity="/scratch/gucr/tEDRAM2/training_data/scene/")
    """

    # path: str = r"./training_data/binocular_image_data.h5"
    # arr = File(path, 'r')

    # left_img: ndarray = arr['binocular_image']['left_img'][1800:2000, :, :]
    # right_img: ndarray = arr['binocular_image']['right_img'][1800:2000, :, :]

    # bi_imsave(left_img, right_img, path_disparity="./training_data/binocular_imgs/")


    """
    filename_dp = r"/training_data/label_data.txt"
    display = DisparityMapsDisplay(getcwd() + filename_dp, filename_bino=getcwd() + path, filename_l=getcwd()+filename_dp)
    display.display_binocular()
    print("it was successfully displayed")
    """
    pass
