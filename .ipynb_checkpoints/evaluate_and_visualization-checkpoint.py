import os
import sys
import numpy as np
sys.path.append('/root/SE/se/RIFE_LSTM_Context')
root = '/root/SE/se/RIFE_LSTM_Context'
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_path = root + "/frames"
output_path = root + "/outputs"
pretrained_model_path = root + '/intrain_log'

if not os.path.exists(frame_path):
    os.mkdir(frame_path)
    
if not os.path.exists(output_path):
    os.mkdir(output_path)

def load_frames(frame_folder, start_frame, num_frames=7):
    # Load a sequence of 'num_frames' starting from 'start_frame'
    frames = []
    for i in range(1,num_frames):
        frame_path = os.path.join(frame_folder, f"frame_{start_frame + i:04d}.png")
        frame = Image.open(frame_path).convert('RGB')
        frames.append(frame)
    '''
    imgs = frames
    h, w =(224,224)
    ih, iw, _ = imgs[0].shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    for i in range(7):
        imgs[i] = imgs[i][x:x+h, y:y+w, :]
        #img1 = img1[x:x+h, y:y+w, :]
        #gt = gt[x:x+h, y:y+w, :]
        #return img0, gt, img1
        return imgs    
    '''
    print('load')
    return frames

def preprocess_frames(frames):
    # Convert frames to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # tensor = torch.stack([transform(frame) for frame in frames], dim=1)  # shape: (3, 7, H, W)
    tensor = torch.stack([transform(frame) for frame in frames], dim=0)  # shape: (3, 7, H, W)
    
    tensor = tensor.view(1, 3*7, tensor.shape[2], tensor.shape[3])  # shape: (1, 21, H, W)
    print('preprocess')
    return tensor.to(device)

def save_frame(tensor, output_folder, frame_index):
    # Save a tensor as an image
    transform = transforms.ToPILImage()
    img = transform(tensor.cpu())
    img.save(os.path.join(output_folder, f"frame_{frame_index:04d}.png"))
    #print('save\n')

def inference_video(model, frame_folder, output_folder, total_frames):
    # assume total_frames is 3x
    with torch.no_grad():
        for start_frame in range(0,total_frames - 3 + 1, 3):  # Adjust the step to handle overlap or gaps
            frames = load_frames(frame_folder, start_frame)
            input_tensor = preprocess_frames(frames)
            print('gointo model')
            output_allframes_tensors = model(input_tensor)
            print('compute finished')
            # Output handling
            #
            # flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_onlyteacher, \
            # Sum_loss_distill, Sum_loss_context, Sum_loss_tea, Sum_loss_ssimd = output

            # Example to extract the interpolated frames
            # Assume output_allframes_tensors is in the shape expected for your outputs
            # Here, just an example of handling the tensor; actual usage may vary
            interpolated_frames = output_allframes_tensors[:-1]  # Skipping first and last if they are not interpolated
            # Assuming the output tensor is ordered in sequence and needs reshaping or reordering
            # Example reshape to (1, 7, 3, H, W)
            print('try save')
            # Saving only interpolated frames: indices 0 to 5. the 6th is the 0th of the next saving loop
            for i in range(6):
                save_frame(interpolated_frames[i, :, :, :], output_folder, start_frame * 2 + i)
            torch.cuda.empty_cache()

from model.inferenceRIFE import Model


model = Model(local_rank=0)
model.load_model(pretrained_model_path )
print("Loaded ConvLSTM model")
model.eval()
model.to_device()
inference_video(model.simple_inference, frame_path, output_path, 366)

start_frame=0
frames = load_frames(frame_path, start_frame)
input_tensor = preprocess_frames(frames)
# Create a writer object
writer = SummaryWriter('runs/Overall')

writer.add_graph(model.flownet, sample_input)