import os
import time

import torch
from torch import nn
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num = -1
    # while num < 713:
    while num < 1701:
        Image.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None
        classes = 1  # exclude background
        weights_path = "./save_weights/best_model.pth"
        num += 1
        name = str(num)
        # name_2 = '2_' + name
        # name_2_01 = name_2 + '.tif'
        # name_2_02 = name_2 + '.tif'
        # img_path = "./all_test\images/" + name_2_01
        name01 = '2_' + name + '.tif'
        name02 = '2_' + name + '.tif'
        img_path = "./all_test/test images/" + name01
    #img_path = "./val_01.tif"
    #roi_mask_path = "./test01_mask.tif"#"./DRIVE/test/mask/01_test_mask.gif"
        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(img_path), f"image {img_path} not found."
    #assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

    # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#device = torch.device("cpu")
        print("using {} device.".format(device))

    # create model
        model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.to(device)

    # load roi mask
    #roi_img = Image.open(roi_mask_path).convert('L')
    #roi_img = np.array(roi_img)

    # load image
        original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
    # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
        # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        #prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            mask.save(name02)
            #mask.save(name_2_02)
        #mask.save("val_01_predict.tif")


if __name__ == '__main__':
    main()
