from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import cv2
import numpy as np
from model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text
import ucf_option
import torch.nn as nn
from crop import video_crop  # 引入裁剪功能


# 初始化 Flask 应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # 设置上传文件的保存路径

# 加载模型和相关参数
device = "cuda" if torch.cuda.is_available() else "cpu"
args = ucf_option.parser.parse_args([])

# 初始化 projection_layer 将视频特征投影到512维度
projection_layer = nn.Linear(150528, 512).to(device)

label_map = {'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'}

model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
model_param = torch.load(args.model_path)
model.load_state_dict(model_param)
model.to(device)
model.eval()



def random_crop(image, crop_size=(224, 224)):
    height, width, _ = image.shape
    print(f"Frame size: {height}x{width}, Crop size: {crop_size}")  # 打印帧和裁剪尺寸

    # 如果帧的大小刚好等于裁剪尺寸，跳过裁剪
    if height == crop_size[0] and width == crop_size[1]:
        print("Frame size matches crop size, skipping crop.")
        return image  # 不进行裁剪

    # 检查帧的大小是否足够大进行裁剪
    if height < crop_size[0] or width < crop_size[1]:
        print(f"Frame size ({height}, {width}) is smaller than crop size {crop_size}, skipping crop.")
        return image  # 如果帧太小，不进行裁剪

    top = np.random.randint(0, height - crop_size[0])
    left = np.random.randint(0, width - crop_size[1])
    cropped_image = image[top:top+crop_size[0], left:left+crop_size[1]]
    return cropped_image

def horizontal_flip(image):
    flipped_image = cv2.flip(image, 1)  # 1表示水平翻转
    return flipped_image

def random_rotation(image, angle_range=(-10, 10)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def random_brightness(image, brightness_range=(0.8, 1.2)):
    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
    bright_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    return bright_image

def augment_frame(image):
    # 随机选择并应用不同的数据增强操作
    if np.random.rand() < 0.5:
        image = random_crop(image)
    if np.random.rand() < 0.5:
        image = horizontal_flip(image)
    if np.random.rand() < 0.5:
        image = random_rotation(image)
    if np.random.rand() < 0.5:
        image = random_brightness(image)
    return image

def augment_video(video_frames):
    augmented_frames = []
    for frame in video_frames:
        augmented_frame = augment_frame(frame)
        augmented_frames.append(augmented_frame)
    return augmented_frames

def video_to_tensor(filepath, maxlen, desired_size=(224, 224)):
    """将视频转换为模型输入的张量格式"""
    cap = cv2.VideoCapture(filepath)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        if frame is None:
            print("Empty frame encountered!")
        else:
            print(f"Read frame with shape: {frame.shape}")
        frame = cv2.resize(frame, desired_size)
        frame = frame / 255.0  # 归一化
        augmented_frame = augment_frame(frame)  # 数据增强
        frames.append(augmented_frame)

    cap.release()

    # 将帧转换为 NumPy 数组，然后再转换为 PyTorch 张量
    frames = np.array(frames).astype(np.float32)
    print(f"Frames shape after loading: {frames.shape}")  # 打印读取后帧的形状
    if frames.shape[0] == 0:
        print("No frames were loaded from the video!")
    frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
    
    if frames_tensor.shape[1] > maxlen:
        frames_tensor = frames_tensor[:, :maxlen]
    print(f"Frames tensor shape before returning: {frames_tensor.shape}")  # 确认最终张量的形状
    return frames_tensor

# 这是假设的 encode_video 内部的代码片段
def encode_video(self, images, padding_mask, lengths):
    # 确保 images 的维度正确
    # 当前的 shape 是 [batch_size, time_steps, channels, height, width]
    
    # 你可能需要将其调整为 [time_steps, batch_size, channels, height, width]
    images = images.permute(1, 0, 2, 3, 4)  # 调整为 [time_steps, batch_size, channels, height, width]

    # 然后，你可以进行其他操作，比如将空间维度展平
    images = images.reshape(images.shape[0], images.shape[1], -1)  # [time_steps, batch_size, channels*height*width]

    # 现在你可以进行 frame_position_embeddings 的加法
    frame_position_embeddings = self.frame_position_embeddings[:images.shape[0], :, :]  # 调整 frame_position_embeddings 的形状
    images = images + frame_position_embeddings

    # 后续其他的处理
    visual_features = self.temporal(images, padding_mask, lengths)
    
    return visual_features


def analyze_video(filepath):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    visual = video_to_tensor(filepath, maxlen=args.visual_length)
    
    print(f"Visual tensor shape before conversion to numpy: {visual.shape}")
    
    visual = visual.numpy()
    print(f"Visual numpy shape: {visual.shape}")
    
    cropped_visual = video_crop(visual, type=0)
    cropped_visual_tensor = torch.tensor(cropped_visual).unsqueeze(0).to(device)  # Ensure tensor is on the same device as projection_layer
    
    print(f"Cropped visual tensor shape after conversion: {cropped_visual_tensor.shape}")
    
    # 检查张量的维度
    if cropped_visual_tensor.dim() == 4:  # 如果是 4 维
        cropped_visual_tensor = cropped_visual_tensor.unsqueeze(0)  # 添加 batch_size 维度
        print(f"Added batch size, new shape: {cropped_visual_tensor.shape}")
    
    # 继续使用
    batch_size, time_steps, channels, height, width = cropped_visual_tensor.shape
    print(f"Batch size: {batch_size}, Time steps: {time_steps}, Channels: {channels}, Height: {height}, Width: {width}")

    # 使用 reshape 将 channels, height, width 展平为一个维度
    visual = cropped_visual_tensor.reshape(batch_size, time_steps, -1).to(device)  # Ensure visual is on the correct device


    # 投影 visual 的最后一个维度到 512
    visual = projection_layer(visual)

    # 生成填充掩码和提示文本
    length = visual.shape[1]
    lengths = torch.tensor([length], dtype=torch.int).to(device)  # 保证 lengths 在同一设备
    padding_mask = get_batch_mask(lengths, args.visual_length).to(device)  # 确保 padding_mask 在设备上
    
    # 使用与 test.py 相同的 get_prompt_text 函数
    prompt_text = get_prompt_text(label_map)  # 确保 prompt_text 在设备上

    # 推理
    with torch.no_grad():
        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)

        # 计算每个类别的概率
        probabilities = logits2.softmax(dim=-1)

        # 对时间步的概率取平均
        avg_probabilities = probabilities.mean(dim=0)  # 先对时间步取平均
        avg_probabilities = avg_probabilities.mean(dim=0)  # 再对类别维度取平均

         # 打印调试信息
        print(f"Per time step probabilities: {probabilities}")
        print(f"Avg Probabilities after mean: {avg_probabilities.shape}, {avg_probabilities}")

        # 获取最大概率的类别索引
        predicted_class = avg_probabilities.argmax(dim=0).item()

    # 动态调整阈值
    predicted_label = list(label_map.values())[predicted_class]
    if predicted_label in ['Abuse', 'Arson', 'Fighting']:
        threshold = 0.05  # 更敏感的异常行为
    else:
        threshold = 0.2  # 普通异常行为

    # 根据阈值判断是否存在异常行为
    if avg_probabilities.max() > threshold and predicted_label!='Normal':
        return f"检测到异常行为: {predicted_label}!"
    else:
        return "没有异常行为"




@app.route('/')
def index():
    """主页面，显示上传表单"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """处理上传视频的请求"""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result = analyze_video(filepath)  # 处理并分析视频
        return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
