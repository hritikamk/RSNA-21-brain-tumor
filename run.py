from torch.nn import functional as F
from torch.utils import data as torch_data
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
import random
import time
from torch import nn
import torch
import numpy as np
import glob
import sys
import os
import pydicom
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import animation, rc
import efficientnet_pytorch
from torch.utils.data import Dataset, DataLoader


rc('animation', html='jshtml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

img_size = 256
n_frames = 10
cnn_features = 256
lstm_hidden = 32
n_fold = 5
n_epochs = 10


class TestDataRetriever(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def read_video(self, vid_paths):
        video = [load_dicom(path) for path in vid_paths]
        if len(video) == 0:
            video = torch.zeros(n_frames, img_size, img_size)
        else:
            video = torch.stack(video)  # T * C * H * W
#         video = torch.transpose(video, 0, 1) # C * T * H * W
        return video

    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = f"../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{str(_id).zfill(5)}/"
        channels = []
        for t in ["FLAIR", "T1w", "T1wCE", "T2w"]:
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, "*")),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            num_samples = n_frames
#             t_paths = get_valid_frames(t_paths)
            if len(t_paths) < num_samples:
                in_frames_path = t_paths
            else:
                in_frames_path = uniform_temporal_subsample(
                    t_paths, num_samples)

            channel = self.read_video(in_frames_path)
            if channel.shape[0] == 0:
                print("1 channel empty")
                channel = torch.zeros(num_samples, img_size, img_size)
            channels.append(channel)

        channels = torch.stack(channels).transpose(0, 1)
        return {"X": channels.float(), "id": _id}


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.net = efficientnet_pytorch.EfficientNet.from_name(
            "efficientnet-b0")
        checkpoint = torch.load(
            'efficientnet-b0-08094119.pth')
        self.net.load_state_dict(checkpoint)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(
            in_features=n_features, out_features=cnn_features)

    def forward(self, x):
        x = F.relu(self.map(x))
        out = self.net(x)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(cnn_features, lstm_hidden, 2, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1, bias=True)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        output, (hn, cn) = self.rnn(r_in)
        out = self.fc(hn[-1])
        return out


def load_dicom(path):
    dicom = pydicom.read_file(path, force=True)
    data = dicom.pixel_array
    data = data-np.min(data)
    if np.max(data) != 0:
        data = data/np.max(data)
    data = (data*255).astype(np.uint8)
    return data


def load_dicom_line(path):
    t_paths = sorted(
        glob.glob(os.path.join(path, "*")),
        key=lambda x: int(x[:-4].split("-")[-1]),
    )
    images = []
    for filename in t_paths:
        data = load_dicom(filename)
        if data.max() == 0:
            continue
        images.append(data)

    return images


def read_video(vid_paths):
    video = [load_dicom(path) for path in vid_paths]
    # video = torch.tensor(video)
    if len(video) == 0:
        video = torch.zeros(n_frames, img_size, img_size)
    # else:
        # video = torch.stack(video)  # T * C * H * W
#         video = torch.transpose(video, 0, 1) # C * T * H * W
    return video


def create_animation(ims):
    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000//24)


def load_model():
    model = Model()
    model.to(device)
    checkpoint = torch.load(f"./best-model-1.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def uniform_temporal_subsample(x, num_samples):
    t = len(x)
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return [x[i] for i in indices]


def predict(predict_path):
    model = load_model()
    patient_path = f"./00001"
    channels = []
    for t in ["FLAIR", "T1w", "T1wCE", "T2w"]:
        t_paths = sorted(
            glob.glob(os.path.join(patient_path, t, "*")),
            key=lambda x: int(x[:-4].split("-")[-1]),
        )
        num_samples = n_frames
#             t_paths = get_valid_frames(t_paths)
        if len(t_paths) < num_samples:
            in_frames_path = t_paths
        else:
            in_frames_path = uniform_temporal_subsample(t_paths, num_samples)

        channel = read_video(in_frames_path)
        channel = torch.tensor(channel)
        if channel.shape[0] == 0:
            print("1 channel empty")
            channel = torch.zeros(num_samples, img_size, img_size)
        channels.append(channel)

    channels = torch.stack(channels).transpose(0, 1)
    model.eval()
    print(channels.shape)
    channels = torch.reshape(
        channels, (1, channels.shape[0], channels.shape[1], channels.shape[2], channels.shape[3]))
    channels = channels.float()
    tmp_res = torch.sigmoid(model(channels.to(device))
                            ).detach().numpy()
    st.write(tmp_res)

    if tmp_res[0][0] > 0.5:
        st.title("brain cancer detected ")
    else:
        st.title("brain cancer not detected in MRI scan ")
    return None


def run_app():

    file_uploaded = st.file_uploader(
        "Upload dicom file folder zipped", accept_multiple_files=True)
    directory = os.getcwd()
    print(directory)
    print(os.listdir(directory))
    # anima = create_animation(load_dicom_line("./00001/FLAIR"))
    #HtmlFile = line_ani.to_html5_video()
    # with open("myvideo.html", "w") as f:
    #     print(anima.to_html5_video(), file=f)
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    HtmlFile = open("myvideo.html", "r")
    # HtmlFile="myvideo.html"
    source_code = HtmlFile.read()
    st.components.v1.html(source_code, height=900, width=900)
    st.balloons()
    predict_path = "./00001"
    prediction = predict(predict_path)
    st.write(prediction)
    return None
