#Image Quality
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
from PIL import Image
from ultralytics import YOLO
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from infer_crackSeg import crack_segmentation

def count_white_pixels(image):
    """
    Count white pixels in an image. Accepts both cv2 and PIL image formats.
    
    Args:
        image: Either a cv2 image array or PIL Image object
        
    Returns:
        int: Number of white pixels in the binary image
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if image has multiple channels
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert int64 image to uint8 format
    image = image.astype(np.uint8)
    
    # Convert to binary using Otsu's method
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Count white pixels (255 values)
    white_pixels = np.sum(binary == 255)
    
    return white_pixels

# The resize and padding function we created earlier
def resize_and_pad_image(image, target_size):
    original_width, original_height = image.size
    
    # Calculate the ratio of target size to original size
    width_ratio = target_size[0] / original_width
    height_ratio = target_size[1] / original_height
    
    # Resize proportionally
    if width_ratio < height_ratio:
        new_width = target_size[0]
        new_height = int(original_height * width_ratio)
    else:
        new_width = int(original_width * height_ratio)
        new_height = target_size[1]
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new image with the target size and fill it with black (or any color)
    new_image = Image.new("RGB", target_size, (0, 0, 0))  # Black background
    
    # Calculate padding (center the image)
    pad_left = (target_size[0] - new_width) // 2
    pad_top = (target_size[1] - new_height) // 2
    
    # Paste the resized image onto the new image
    new_image.paste(resized_image, (pad_left, pad_top))
    
    return new_image

def ImageQuality(image):
    '''
    input: image(opencv)

    result_state = [blurness, brightness, color]
    blurness = {0: too blur; 1: passed}
    blurness = {0: too bright; 1: too dark; 2: passed}
    color = {0: color deviation; 1: passed}

    return: booling, result_state(list)
    ex: 
    import cv2
    image = cv2.imread('path/to/img')
    result, result_state = ImageQuality(image)

    '''
    result_state = []
    ########### blurness ###########

    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F)
    variance = np.var(imageVar)
        # 显示原始图像和拉普拉斯图像
    # plt.figure(figsize=(6, 6))

    # plt.imshow(imageVar, cmap='gray')
    # plt.title('Laplacian')
    # plt.axis('off')

    # plt.show()
    #print(variance)
    if variance < 250:
        result_state.append(0)

    else:
        result_state.append(1)

    ########### brightness ###########
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰階直方圖展示
    # grayImg = gray_img.copy()
    # plt.hist(grayImg.ravel(), 256)
    # plt.show()
    # height, width = gray_img.shape
    height, width = gray_img.shape
    size = gray_img.size
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = np.sum(shift_value)
    da = shift_sum / size
    ma = 0

    for i in range(256):
        ma += (abs(i-128-da) * hist[i])
    m = abs(ma / size)
    k = abs(da) / m
    #print(k)
    #print(da)
    if k[0] > 1:
        if da > 0:
            result_state.append(0) # too bright
        else:
            result_state.append(1) # too dark

    else:
        result_state.append(2)

    ########### color ###########
    img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img)
    h, w,_  = img.shape
    da = a_channel.sum() / (h * w) - 128
    db = b_channel.sum() / (h * w) - 128
    histA = [0] * 256
    histB = [0] * 256

    for i in range(h):
        for j in range(w):
            ta = a_channel[i][j]
            tb = b_channel[i][j]
            histA[ta] += 1
            histB[tb] += 1

    msqA = msqB = 0

    for y in range(256):
        msqA += float(abs(y - 128 - da)) * histA[y] / (w * h)
        msqB += float(abs(y - 128 - db)) * histB[y] / (w * h)

    result = math.sqrt(da * da + db * db) / math.sqrt(msqA * msqA + msqB * msqB)

    if result > 1.5:
        result_state.append(0)

    else:
        result_state.append(1)

    if result_state[0] == 0 or result_state[1] != 2 and result_state[2] !=1 :
        return (False, result_state)
    else:
        return (True, result_state)


     

def DamageClassification(img, model):
    """
    modelType: 
    wall / column / beam

    return: string
    Class A / Class B / Class C

    """
    #model = ViTForImageClassification.from_pretrained(f'classificationTrain/models/old/{detectionType}')
    model.eval()
    size = (384, 384)
    id2label =  {0: 'Class A', 1: 'Class B', 2: 'Class C'}
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=image_mean, std=image_std),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    
    # Load and process the image
    #img = Image.open(img_path).convert("RGB")

    resized_img = resize_and_pad_image(img, size)
    transformed_img = transform(resized_img).unsqueeze(0)

    # Get model predictions
    with torch.no_grad():
        outputs = model(transformed_img)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_prob, pred_class = torch.max(probs, dim=-1)
        pred_label = id2label[pred_class.item()]
    return pred_label



def Infer_spallingSeg(model, img ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def predict(image):
        image = image.unsqueeze(0)
        h, w = image.shape[-2:]
        #image = nn.functional.interpolate(image,
        #            size=[864, 864], # (height, width)
        #            mode='bilinear',
        #            align_corners=False)
        image=image.to(device)
        with torch.no_grad():
            output = model(image)[1]
            output = torch.argmax(output, dim=1)
            output = output.cpu().squeeze(0).numpy()
            #output = cv2.resize(output, [w, h], interpolation=cv2.INTER_NEAREST)
        return output
    img = np.array(img) # PIL to numpy for cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.Resize(864, 864), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()])
    aug = transform(image=img)
    img = aug['image']
    pred = predict(img)
    return(pred * 255)




def DamageDetection(img,image_cv, model_detection,model_crackClassification,detection_type,CE,ratio, model_crackSeg, model_spallingSeg):
    
    """
    modelType: 
    one detection model

    return: list
    [Leakage of rebar, Major-Crack, Minor-crack, Spalling]

    """

    total_cost_crack = 0
    total_cost_spalling = 0
    total_cost_rebar = 0

    # unit pixel length in reality
    ratio = ratio*ratio # cm^2
    cost_spalling = 7000 # 一坪 7000 (工法 + 磁磚)
    
    
    original_img=copy.deepcopy(image_cv)
    # configuration for crack classfication model
    # model_crackClassification.eval()
    std_size = (384, 384)
    id2label_mapping = {
    'wall': None,
    'column': None,
    'beam': None
    }
    id2label_mapping['wall'] =  {0: 'Diagonal', 1: 'Diagonal_large', 2: 'Horizontal', 3:'Horizontal_large', 4:'spalling-like_cracks', 5:'Vertical', 6:'Vertiacal_large', 7:'Web',8: 'Web_large', 9:'X-shape'}
    id2label_mapping['column']  =  {0: 'Diagonal', 1: 'Horizontal', 2:'Horizontal_large', 3:'Vertical', 4:'Vertiacal_large', 5:'Web', 6:'X-shape'}
    id2label_mapping['beam']  =  {0: 'Diagonal', 1: 'Horizontal', 2:'Vertical', 3:'Web'}

    crackPrice_mapping = {
    'wall': None,
    'column': None,
    'beam': None
    }
    crackPrice_mapping['wall'] =  {'Diagonal': 511,'Diagonal_large':1705, 'Horizontal':511, 'Horizontal_large':1705, 'spalling-like_cracks':511 ,'Vertical':511 , 'Vertiacal_large':1705, 'Web':511, 'Web_large':1705, 'X-shape':1705}
    crackPrice_mapping['column']  =  {'Diagonal':25831,'Horizontal':135, 'Horizontal_large':25831, 'Vertical':1198, 'Vertiacal_large':23219, 'Web':25831, 'X-shape':25831}
    crackPrice_mapping['beam']  =  {'Diagonal':33347, 'Horizontal':1728, 'Vertical':1728, 'Web':33347}

    dimension_assumption = 0
    if detection_type == 'wall':
        dimension_assumption = 90000 # cm^2
    elif detection_type == 'column':
        dimension_assumption = 13500 # cm^2
    if detection_type == 'beam':
        dimension_assumption = 10000 # cm^2


    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=image_mean, std=image_std),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    #model = YOLO('runs/detect/9e/weights/best.pt')
    #img = cv2.imread("datasets/test/images/cracking217_jpg.rf.03ff1acc26afb21d3be1f1eed68f5c0c.jpg")
    print(f"orginal image size{img.size}")
    # YOLO use (640,448) as input size
    results = model_detection(img, conf = 0.2)
 

    boxes = results[0].boxes
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    #confidences = results[0].boxes.conf.tolist()
    total_result = []
    boxIndex = 0
    for box in boxes: # {0: 'Expose of rebar', 1: 'Cracks', 2: 'Spalling'}
        boxIndex +=1 
        # crack classfication
        class_id = int(box.cls[0])
        if class_id == 1:  # Check if the class is "1" which is crack
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Crop using PIL
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img.convert('RGB')
            cropped_image = np.array(cropped_img)
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'./modelTest/resultImages/originalCropped{boxIndex}.jpg', cropped_image)
            resized_img = resize_and_pad_image(cropped_img, std_size)
            transformed_img = transform(resized_img).unsqueeze(0)
            
            # Get model predictions
            with torch.no_grad():
                outputs = model_crackClassification(transformed_img)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_prob, pred_class = torch.max(probs, dim=-1)
                id2label = id2label_mapping[detection_type]
                pred_label = id2label[pred_class.item()]

            if pred_label in total_result:
                pass
            else:
                total_result.append(pred_label)
            
            # Draw bounding box and label on the original image
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_cv, f"{results[0].names[class_id]}, conf:{box.conf[0]:.2f}",
                        (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            if CE:
                # cost estimation for this cropped crack image
                
                ori_size = (cropped_image.shape[1], cropped_image.shape[0])  # (width, height)
                # resize the cropped image to the original size by the propotion of the first resize(orignal to 312)

                print(f"original size: {cropped_image.shape}")
                patch_result, full_result = crack_segmentation(cropped_img,model_crackSeg)
                result_crack_seg = (full_result * 255).astype(np.uint8)
                cv2.imwrite(f'./modelTest/resultImages/crackSeg{boxIndex}.jpg', result_crack_seg)
                result_crack_seg = cv2.resize(result_crack_seg, ori_size)
                print(f"re-size after crack seg: {result_crack_seg.shape}")
                white = count_white_pixels(result_crack_seg)

                # cost for crack
                crackLabel =  crackPrice_mapping[detection_type]
                cost_crack = crackLabel[pred_label]

                total_cost_crack += white * ratio / dimension_assumption * cost_crack
            

        elif class_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            total_result.append('Expose of rebar')
            # Draw bounding box and label on the original image
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_cv, f"Expose of rebar, conf:{box.conf[0]:.2f}",
                        (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            
            # cost estimation for rebar
            if CE:
                if detection_type == 'wall':
                    cost_rebar = 38743 
                elif detection_type == 'column':
                    cost_rebar = 23219
                if detection_type == 'beam':
                    cost_rebar = 29280 
                total_cost_rebar += cost_rebar

        elif class_id == 2:
            total_result.append('Spalling')
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Crop using PIL
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img.convert('RGB')
            cropped_image = np.array(cropped_img)
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'./modelTest/resultImages/originalCropped{boxIndex}.jpg', cropped_image)
            if CE:
                # cost estimation for this cropped spalling image
                ori_size = (cropped_image.shape[1], cropped_image.shape[0])  # (width, height)
                # resize the cropped image to the original size by the propotion of the first resize(orignal to 312)

                print(f"original size: {cropped_image.shape}")
                result_spalling_seg = Infer_spallingSeg(model_spallingSeg, cropped_img)
                result_spalling_seg = result_spalling_seg.astype(np.uint8)
                result_spalling_seg = cv2.resize(result_spalling_seg, ori_size, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f'./modelTest/resultImages/spallingSeg{boxIndex}.jpg', result_spalling_seg)
                print(f"re-size after spalling seg: {result_spalling_seg.shape}")
                white = count_white_pixels(result_spalling_seg)
                total_cost_spalling += white * ratio* cost_spalling / 32400 # 每坪32400 cm^2
            
            # Draw bounding box and label on the original image
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_cv, f"Spalling, conf:{box.conf[0]:.2f}",
                        (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)







    # Print detected classes
    detection_result = list(set(results[0].names[int(box.cls[0])] for box in boxes))
    print("Detected classes:", detection_result)
    print("classification result:", total_result)
    rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'./modelTest/resultImages/detectResult.jpg', rgb_image)
    detected_img = Image.fromarray(rgb_image)

    return total_result, original_img, detected_img, total_cost_crack,total_cost_spalling, total_cost_rebar

def checkReason(detectionType,classify_result,total_result):
    reason = 'no siginificant reason detected'
    reasonList = []

    #[Expose of rebar, Major-Crack, Minor-crack, Spalling]
    if detectionType == "wall":

        if 'Expose of rebar' in total_result:
            classify_result = 'Class A'
        elif 'X-shape' in total_result:
            classify_result = 'Class A'
        elif 'Web_large' in total_result:
            classify_result = 'Class A'
        


        if classify_result == 'Class A':
            if 'Expose of rebar' in total_result:
                reasonList.append('Expose of rebar')

            if 'Spalling'in total_result:
                reasonList.append('Spalling')

            if 'X-shape' in total_result:
                reasonList.append('X-shape crack')

            if 'Web_large' in total_result:
                reasonList.append('Large Web-like crack which might contain shear or huge spalling crack')

        elif classify_result == 'Class B':
            if 'Diagonal_large' in total_result:
                reasonList.append('Large diagonal crack')
            if 'Horizontal_large' in total_result:
                reasonList.append('Large Horizontal crack')
            if 'Vertical_large' in total_result:
                reasonList.append('Large Vertical crack')
            if 'Spalling'in total_result:
                reasonList.append('Spalling')
            if 'spalling-like_cracks'in total_result:
                reasonList.append('spalling-like cracks')

        elif classify_result == 'Class C':
            if 'Diagonal' in total_result:
                reasonList.append('Small diagonal crack')
            if 'Horizontal' in total_result:
                reasonList.append('Small Horizontal crack')
            if 'Vertical' in total_result:
                reasonList.append('Small Vertical crack')
            if 'Web' in total_result:
                reasonList.append('Small Web-like cracks')
            if 'Spalling'in total_result:
                reasonList.append('Small Spalling')
            if 'spalling-like_cracks'in total_result:
                reasonList.append('spalling-like cracks')
            


    elif detectionType == "column":
        if 'Expose of rebar' in total_result or 'X-shape' in total_result or "Diagonal" in total_result or "Web" in total_result or "Vertical_large" in total_result:
            classify_result = 'Class A'

        if classify_result == 'Class A':
            if 'Expose of rebar' in total_result:
                reasonList.append('Expose of rebar')

            if 'Spalling'in total_result:
                reasonList.append('Spalling')

            if 'X-shape' in total_result:
                reasonList.append('X-shape crack')

            if 'Diagonal' in total_result:
                reasonList.append('Diagonal crack')

            if 'Web' in total_result:
                reasonList.append('Web-like cracks')

            if 'Vertical_large' in total_result:
                reasonList.append('Large Vertical crack')

        if classify_result == 'Class B':
            if 'Horizontal_large' in total_result:
                reasonList.append('Large Horizontal crack')
            if 'Vertical' in total_result:
                reasonList.append('Small Vertical crack')
            if 'Spalling'in total_result:
                reasonList.append('Small Spalling')


        if classify_result == 'Class C':
            if 'Horizontal' in total_result:
                reasonList.append('Small Horizontal crack')
            if 'Spalling'in total_result:
                reasonList.append('Small Spalling')    


    elif detectionType == "beam":
        if 'Expose of rebar' in total_result:
            classify_result = 'Class A'

        if classify_result == 'Class A':
            if 'Expose of rebar' in total_result:
                reasonList.append('Expose of rebar')


        if classify_result == 'Class B':
            if 'Spalling'in total_result:
                reasonList.append('Spalling')

            if 'Diagonal' in total_result:
                reasonList.append('Diagonal crack')

            if 'Web' in total_result:
                reasonList.append('Web-like cracks')

            if 'Horizontal' in total_result:
                reasonList.append('Horizontal crack')

        if classify_result == 'Class C':
            if 'Vertical' in total_result:
                reasonList.append('Small Vertical crack')
    if reasonList:
        reason = " and ".join(reasonList)  # For all items
        reason = reason + ' detected.'
    ClassType = classify_result
    return ClassType, reason
    




