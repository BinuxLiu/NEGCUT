import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util

def list_all_subdirectories(folder_path):
    subdirectories = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subdirectories.append(os.path.join(root, dir_name))
    return subdirectories

def list_images_in_directory(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(os.path.join(directory, file))
    return image_files


class VPRDataset(BaseDataset):
    """
    This dataset class can load vpr datasets.

    It requires one directories to host vpr images from domain day.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dirs = list_all_subdirectories(opt.dataroot)

        self.paths = []
        for dir in self.dirs:
            if dir.split("/")[-1] != "database":
                self.paths.extend(list_images_in_directory(dir))
        self.paths = sorted(self.paths)
        self.size = len(self.paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        path = self.paths[index % self.size]  # make sure index is within then range

        img = Image.open(path).convert('RGB')

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(img)

        return {'A': A, 'B': A, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.size
