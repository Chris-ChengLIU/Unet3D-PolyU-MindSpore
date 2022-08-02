# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import numpy as np
from matplotlib import pyplot as plt
from mindspore import dtype as mstype
from mindspore import Model, context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
# from src.dataset import create_dataset
from src.unet3d_model import UNet3d, UNet3d_
from src.utils import create_sliding_window, CalculateDice
from src.transform import Dataset, ExpandChannel, LoadData, Orientation, ScaleIntensityRange, RandomCropSamples, OneHot
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

import mindspore.dataset as ds
# from mindspore.dataset.transforms.transforms import Compose
from mindspore.dataset.transforms.c_transforms import Compose
from pathlib import Path


import pydicom
from skimage import measure
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id=device_id)




class ConvertLabel:
    """
    Crop at the center of image with specified ROI size.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
        If its components have non-positive values, the corresponding size of input image will be used.
    """
    def operation(self, data):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        data[data > config.upper_limit] = 0
        data = data - (config.lower_limit - 1)
        data = np.clip(data, 0, config.lower_limit)
        return data

    def __call__(self, image, label):
        label = self.operation(label)
        return image, label

def create_dataset(data_path, seg_path=None, rank_size=1, rank_id=0, is_training=True):
    seg_files = [data_path]
    train_files = [data_path]
    train_ds = Dataset(data=train_files, seg=seg_files)
    train_loader = ds.GeneratorDataset(train_ds, column_names=["image", "seg"], num_parallel_workers=1, \
                                       shuffle=is_training, num_shards=rank_size, shard_id=rank_id)

    if is_training:
        transform_image = Compose([LoadData(),
                                   ExpandChannel(),
                                   Orientation(),
                                   ScaleIntensityRange(src_min=config.min_val, src_max=config.max_val, tgt_min=0.0, \
                                                       tgt_max=1.0, is_clip=True),
                                   RandomCropSamples(roi_size=config.roi_size, num_samples=2),
                                   ConvertLabel(),
                                   OneHot(num_classes=config.num_classes)])
    else:
        transform_image = Compose([LoadData(),
                                   ExpandChannel(),
                                   Orientation(),
                                   ScaleIntensityRange(src_min=config.min_val, src_max=config.max_val, tgt_min=0.0, \
                                                       tgt_max=1.0, is_clip=True),
                                   ConvertLabel()])

    train_loader = train_loader.map(operations=transform_image, input_columns=["image", "seg"], num_parallel_workers=1,
                                    python_multiprocessing=False)
    if not is_training:
        train_loader = train_loader.batch(1)
    return train_loader


def get_list_of_files_in_dir(directory, file_types='*'):
    """
    Get list of certain format files.

    Args:
        directory (str): The input directory for image.
        file_types (str): The file_types to filter the files.
    """
    return [f for f in Path(directory).glob(file_types) if f.is_file()]

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass

def save_figure(imgs, mask, slice_index=80, save_path='./', save_times=0):
    """
    Save inference (comparison) figure.

    :param imgs: The 4-dim array data for inference (ndarray)
    :param mask: The mask of testing; 4-dim array (ndarray)
    :param slice_index: The dicom slice index need to be viewed; default is 80 (int, optional)
    :param save_path: The path for saving predicted result (string, optional)
    :return: None
    """
    os.makedirs(os.path.join(save_path, str(save_times)), exist_ok=True)


    figkword = 'sliceidx'
    figsn = str(slice_index)
    figname = figkword + figsn

    sliceidx = slice_index
    a = rescale_intensity(imgs[sliceidx, :, :, 0], out_range=(0, 1))
    b = (mask[sliceidx, :, :, 0]).astype('uint8')

    fig, subplt = plt.subplots(1, 2)
    subplt[0].imshow(mark_boundaries(a, b))
    subplt[1].imshow(b)

    # if save_path == './':
    #     plt.savefig(save_path + figname)
    # elif save_path != './':
    #     mkdir(save_path)
    #     plt.savefig(save_path + figname)
    # plt.savefig(save_path + figname)
    plt.savefig(os.path.join(os.path.join(save_path, str(save_times)), figname))
    plt.clf()
    plt.close()


@moxing_wrapper()
def infer_net(data_path, ckpt_path):
    data_dir = get_list_of_files_in_dir(data_path, '*.nii.gz')
    eval_dataset = create_dataset(data_path=str(data_dir[0]), is_training=False)
    eval_data_size = eval_dataset.get_dataset_size()
    print("infer dataset length is:", eval_data_size)

    if config.device_target == 'Ascend':
        network = UNet3d()
    else:
        network = UNet3d_()
    network.set_train(False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    model = Model(network)
    index = 0
    total_dice = 0
    for batch in eval_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = batch["image"]
        # seg = batch["seg"]
        print("current image shape is {}".format(image.shape), flush=True)
        sliding_window_list, slice_list = create_sliding_window(image, config.roi_size, config.overlap)
        image_size = (config.batch_size, config.num_classes) + image.shape[2:]
        output_image = np.zeros(image_size, np.float32)
        count_map = np.zeros(image_size, np.float32)
        importance_map = np.ones(config.roi_size, np.float32)
        for window, slice_ in zip(sliding_window_list, slice_list):
            window_image = Tensor(window, mstype.float32)
            pred_probs = model.predict(window_image)
            output_image[slice_] += pred_probs.asnumpy()
            count_map[slice_] += importance_map
        output_image = output_image / count_map  # (1, 4, 512, 512, 199)
        # dice, _ = CalculateDice(output_image, seg)
        # print("The {} batch dice is {}".format(index, dice), flush=True)
        # total_dice += dice
        index = index + 1

        # save the segmentation result of all slices
        savepath = './output/'
        os.makedirs(savepath, exist_ok=True)
        mask_test = np.transpose(output_image, [1,4,2,3,0]) # for saving, convert to (4, 199, 512, 512, 1)
        img_test = np.transpose(image, [1,4,2,3,0]) # # for saving, convert to (1, 199, 512, 512, 1)
        # print(mask_test.shape)
        # print(img_test.shape)
        for i in range(mask_test.shape[0]):  # 4 times to save 4 classes
            print('Testing')
            for j in range(mask_test.shape[1]):  # save each slices, e.g. 199
                # save_figure(batch[i], mask_test, slice_index=j, save_path=savepath)
                save_figure(img_test[0], mask_test[i], slice_index=j,
                            save_path=savepath, save_times=i)
            print('\n')


    # avg_dice = total_dice / eval_data_size
    print("**********************End Infer***************************************")
    # print("eval average dice is {}".format(avg_dice))


if __name__ == '__main__':
    # bash ./run_standalone_infer_gpu_fp32.sh ../LUNA16/inference/ ../Unet3d-10_877.ckpt
    infer_net(data_path=config.data_path,
              ckpt_path=config.checkpoint_file_path)
    # infer_net(data_path='LUNA16/inference/',
    #           ckpt_path='Unet3d-10_877.ckpt')
