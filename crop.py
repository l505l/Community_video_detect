import cv2
import numpy as np

import torch
from models import clip
from PIL import Image

def video_crop(video_frame, type):
    l = video_frame.shape[0]
    new_frame = []
    for i in range(l):
        print(f"Frame {i} shape before any processing: {video_frame[i].shape}")
        # 如果是四维 (256, 3, 224, 224)，我们处理时间帧
        if len(video_frame[i].shape) == 4 and video_frame[i].shape[0] == 256:
            for t in range(256):  # 遍历每个时间帧
                frame = np.transpose(video_frame[i][t], (1, 2, 0))  # 转换为 (224, 224, 3)
                print(f"Frame {i} at time {t} shape after transpose: {frame.shape}")
        
                # 在调用 cv2.resize 之前进行详细检查
                if frame is None or frame.size == 0:
                    print(f"Frame {i} at time {t} is empty or invalid.")
                    continue

                # 调用 cv2.resize 之前检查维度和数据
                print(f"Processing frame {i} at time {t}, shape: {frame.shape}")
                try:
                    img = cv2.resize(frame, dsize=(340, 256))  # 调整尺寸
                    new_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    print(f"Error resizing frame {i} at time {t}: {e}")
                    continue
        else:
            print(f"Unexpected frame shape: {video_frame[i].shape}")

    #1
    img = np.array(new_frame)
    if type == 0:
        img = img[:, 16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:, :224, :224, :]
    #3
    elif type == 2:
        img = img[:, :224, -224:, :]
    #4
    elif type == 3:
        img = img[:, -224:, :224, :]
    #5
    elif type == 4:
        img = img[:, -224:, -224:, :]
    #6
    elif type == 5:
        img = img[:, 16:240, 58:282, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #7
    elif type == 6:
        img = img[:, :224, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #8
    elif type == 7:
        img = img[:, :224, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #9
    elif type == 8:
        img = img[:, -224:, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #10
    elif type == 9:
        img = img[:, -224:, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    
    return img

def image_crop(image, type):
    img = cv2.resize(image, dsize=(340, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #1
    if type == 0:
        img = img[16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:224, :224, :]
    #3
    elif type == 2:
        img = img[:224, -224:, :]
    #4
    elif type == 3:
        img = img[-224:, :224, :]
    #5
    elif type == 4:
        img = img[-224:, -224:, :]
    #6
    elif type == 5:
        img = img[16:240, 58:282, :]
        img = cv2.flip(img, 1)
    #7
    elif type == 6:
        img = img[:224, :224, :]
        img = cv2.flip(img, 1)
    #8
    elif type == 7:
        img = img[:224, -224:, :]
        img = cv2.flip(img, 1)
    #9
    elif type == 8:
        img = img[-224:, :224, :]
        img = cv2.flip(img, 1)
    #10
    elif type == 9:
        img = img[-224:, -224:, :]
        img = cv2.flip(img, 1)
    
    return img

if __name__ == '__main__':
    video = np.zeros([3, 320, 240, 3], dtype=np.uint8)
    corp_video = video_crop(video, 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device)
    video_features = torch.zeros(0).to(device)
    with torch.no_grad():
        for i in range(video.shape[0]):
            img = Image.fromarray(corp_video[i])
            img = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img)
            video_features = torch.cat([video_features, feature], dim=0)
    
    video_features = video_features.detach().cpu().numpy()
    np.save('save_path', video_features)