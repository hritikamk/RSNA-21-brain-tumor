import streamlit as st
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from introd import *
from about import *
from run import *


st.title('RSNA-21 Brain tumor detection')

x = st.sidebar.selectbox("choose app mode", ['Introduction', 'RUN >', 'ABOUT'])

if x == 'Introduction':
    intro()
elif x == 'RUN >':
    run_app()
elif x == 'ABOUT':
    about()
