import os
import sys
import numpy as np
sys.path.append('/root/SE/se/RIFE_LSTM_Context')
root = '/root/SE/se/RIFE_LSTM_Context'
import torch
import torchvision.transforms as transforms
from PIL import Image

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_path = root + "/frames"
output_path = root + "/outputs/outputs_real"
pretrained_model_path = root + '/intrain_log'
pretrained_path = root + 'RIFE_log' # pretrained RIFE path
shift = 0



if not os.path.exists(frame_path):
    os.mkdir(frame_path)
    
if not os.path.exists(output_path):
    os.mkdir(output_path)

def load_frames(frame_folder, start_frame, num_frames=4):
    # Load a sequence of 'num_frames' starting from 'start_frame'
    frames = []
    for i in range(1,num_frames+1):
        frame_path = os.path.join(frame_folder, f"frame_{start_frame + i:04d}.png")
        frame = Image.open(frame_path).convert('RGB')
        frames.append(frame)
        if i != num_frames:
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
    tensor = torch.stack([transform(frame) for frame in frames], dim=0) # shape: (3, 7, H, W)
    print("Tensor.shape", tensor.shape)
    tensor = tensor.view(1, 3*7, tensor.shape[2], tensor.shape[3])  # shape: (1, 21, H, W)
    
    print('preprocess')
    return tensor.to(device)

def save_frame(tensor, output_folder, frame_index):
    transform = transforms.ToPILImage()
    img = transform(tensor.cpu())
    img.save(os.path.join(output_folder, f"frame_{frame_index:04d}.png"))
    #print('save\n')

def inference_video(model, frame_folder, output_folder, total_frames):
    with torch.no_grad():
        for start_frame in range(0,total_frames - 3 + 1, 4):  # Adjust the step to handle overlap or gaps

            # manual shift
            start_frame += shift
            i = int(start_frame/4)
            frames = load_frames(frame_folder, start_frame)
            save_start_point = i*6
            input_tensor = preprocess_frames(frames)
            print('gointo model')
            output_allframes_tensors = model(input_tensor)
            print('compute finished')
            interpolated_frames = output_allframes_tensors[:-1] 
            # Example reshape to (1, 7, 3, H, W)
            print('try save')
            # Saving only interpolated frames: indices 0 to 5. the 6th is the 0th of the next saving loop
            for i in range(6):
                save_frame(interpolated_frames[i, :, :, :], output_folder, save_start_point + i + 1)
            torch.cuda.empty_cache()


def convertload(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }


class IFNet_update(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)
        # self.contextnet = Contextnet()
        # self.unet = Unet()

    def forward(self, x, scale_list=[4, 2, 1]):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        loss_cons = 0
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1),
                              torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        '''
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, 1:4] * 2 - 1
        '''
        for i in range(3):
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            # merged[i] = torch.clamp(merged[i] + res, 0, 1)
        return flow_list, mask_list[2], merged


from model.inferenceRIFE import Model

# Load pretrained Optical Flow Model
checkpoint = convertload(torch.load(f'{pretrained_path}/flownet.pkl', map_location=device))
Ori_IFNet_loaded = IFNet_update()
Ori_IFNet_loaded.load_state_dict(checkpoint)
for param in Ori_IFNet_loaded.parameters():
    param.requires_grad = False

model = Model(Ori_IFNet_loaded, local_rank=0)
model.load_model(pretrained_model_path )
print("Loaded ConvLSTM model")
model.eval()
model.to_device()
inference_video(model.simple_inference, frame_path, output_path, 16)
