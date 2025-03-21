import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from crackSeg_unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
from PIL import Image


def crack_segmentation(cropped_img, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Fixed parameters
    threshold = 0.2


    # Preprocessing
    channel_means = [0.485, 0.456, 0.406]
    channel_stds = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

    def evaluate_img(img):
        input_width, input_height = input_size[0], input_size[1]

        img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
        X = train_tfms(Image.fromarray(img_1))
        X = Variable(X.unsqueeze(0)).to(device)
        #X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

        mask = model(X)

        mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
        mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
        return mask

    def evaluate_img_patch(img):
        input_width, input_height = input_size[0], input_size[1]

        img_height, img_width, img_channels = img.shape

        if img_width < input_width or img_height < input_height:
            return evaluate_img(img)

        stride_ratio = 0.1
        stride = int(input_width * stride_ratio)

        normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

        patches = []
        patch_locs = []
        for y in range(0, img_height - input_height + 1, stride):
            for x in range(0, img_width - input_width + 1, stride):
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))

        patches = np.array(patches)
        if len(patch_locs) <= 0:
            return None

        preds = []
        for i, patch in enumerate(patches):
            patch_n = train_tfms(Image.fromarray(patch))
            X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
            masks_pred = model(X)
            mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
            preds.append(mask)

        probability_map = np.zeros((img_height, img_width), dtype=float)
        for i, response in enumerate(preds):
            coords = patch_locs[i]
            probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

        return probability_map


    # Load and preprocess image
    img_0 = cropped_img
    #img_0 = Image.open(str(cropped_img))
    img_0 = np.asarray(img_0)
    if len(img_0.shape) != 3:
        raise ValueError(f'Incorrect image shape:{img_0.shape}')
    img_0 = img_0[:,:,:3]
    img_height, img_width, img_channels = img_0.shape

    # Process full image
    prob_map_full = evaluate_img(img_0)

    # Process patched image
    
    if img_0.shape[0] > 2000 or img_0.shape[1] > 2000:
        img_1 = cv.resize(img_0, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
    else:
        img_1 = img_0
    prob_map_patch = evaluate_img_patch(img_1)

    # Prepare visualization for patch result
    prob_map_viz_patch = prob_map_patch.copy()
    prob_map_viz_patch = prob_map_viz_patch / prob_map_viz_patch.max()
    prob_map_viz_patch[prob_map_viz_patch < threshold] = 0.0

    # Prepare visualization for full result
    #prob_map_viz_full = prob_map_full.copy()
    #prob_map_viz_full[prob_map_viz_full < threshold] = 0.0

    return prob_map_viz_patch, prob_map_full

# Example usage:
#model_path_seg = './models/model_unet_vgg_16_best.pt'
#cropped_img = "./test_result_main/origina_cropped_0.jpg"
#patch_result, full_result = crack_segmentation(cropped_img, model_path_seg)
#cv.imwrite('./test_result_main/single_test.jpg', (full_result * 255).astype(np.uint8))
