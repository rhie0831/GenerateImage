# Crate new image using pkl file generated from StyleGAN2-pytorch
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
import dnnlib as dnnlib
from dnnlib import tflib 
import random
from pathlib import Path
from PIL import Image
import pickle
import numpy as np
import torch
import ipywidgets as widgets
from IPython.display import display
from tqdm import tqdm
from ipywidgets import interact
from ipywidgets import HBox, VBox, Layout
import matplotlib.pyplot as plt
import projector_rev
import csv

def load_model(pklfilename):
    path = pklfilename
    with open(path, 'rb') as f:
        Gs = pickle.load(f)['G_ema'].cuda() # torch.nn.Module

    noise_vars = {name: buf for (name, buf) in Gs.synthesis.named_buffers() if 'noise_const' in name}
    Gs_kwargs = dnnlib.util.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    return Gs, noise_vars, Gs_kwargs

def calculate_latent_vectors(Gs, noise_vars, seeds):
    sum_avg = 0
    n = len(seeds)
    alpha = 1 / n
    for i in range(n):
        img, latent_code = generate_image_random(Gs, noise_vars, seeds[i])
        avg = latent_code * alpha
        sum_avg += avg

    averaged_code = sum_avg

    return averaged_code

def generate_average_image(rndnumbers, adj, Ga, noise_vars):
#     Gs, noise_vars, Gs_kwargs = load_model(pklfilename)
    adj_folder = adj + '/'
    seeds = []

    for rndnumber in rndnumbers:
        if rndnumber is not '':
            seeds.append(int(rndnumber))

    averaged_code = calculate_latent_vectors(Gs, noise_vars, seeds)
    c = None
    image_avg = Gs(averaged_code, c)
    img = (image_avg.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    Image.fromarray(img[0].cpu().numpy(), 'RGB').save('1. chair_image/' + adj_folder + adj + '.png')

def generate_image_random(Gs, noise_vars, rand_seed):
    
    z = torch.from_numpy(np.random.RandomState(rand_seed).randn(1, Gs.z_dim)).to(device)
    w = Gs.mapping(torch.from_numpy(z).to(device), None)
    w = w[:, :1, :].cpu().numpy().astype(np.float32)
    c = None # category

    images = Gs.synthesis(w, noise_mode='const')

    return images, z
    
def generate_image_from_projected_latents(latent_vector):
    image = Gs.synthesis(latent_vector)    
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = Image.fromarray(image[0].cpu().numpy(), 'RGB')
    
    return image

def save_images(self):   
    modified_latent_code = apply_latent_controls_prev()
    image = generate_image_from_projected_latents(modified_latent_code)
    counter = 1    
    path = "4. user_final_image/final_image_chair.jpeg"
    while os.path.exists(path):
        path = "4. user_final_image/final_image_chair(" + str(counter) + ").jpeg"
        counter += 1
    image.save(path)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def apply_latent_controls_prev():
    image_outputs = controller.children[0]
    feature_sliders = controller.children[1]
    
    slider_hboxes = feature_sliders.children[:-2]
    latent_movements = [x.children[0].value for x in slider_hboxes[:-1]]

    all_w1 = torch.tensor(latent_code_list[0], device=device)
    w_avg = ws_image_to_use
    modified_latent_code = w_avg + ((all_w1 - w_avg) * ((latent_movements[0])/100))
    
    for x in range(1, len(latent_code_list)):
        all_wx = torch.tensor(latent_code_list[x], device=device)    
        w_avg = ws_image_to_use
        modified_latent_code += ((all_wx - w_avg) * ((latent_movements[x])/100))
    
    return modified_latent_code

def apply_latent_controls(self):   
    modified_latent_code = apply_latent_controls_prev()
    modified_img = generate_image_from_projected_latents(modified_latent_code)
    image_outputs = controller.children[0]
    latent_img_output = image_outputs.children[1]
    
    with latent_img_output:
        latent_img_output.clear_output()
        display(modified_img)
    
def reset_latent_controls(self):
    image_outputs = controller.children[0]
    feature_sliders = controller.children[1]
    
    slider_hboxes = feature_sliders.children[:-2]
    for x in slider_hboxes[:-1]:
        x.children[0].value = 0
        
    latent_img_output = image_outputs.children[1]
    with latent_img_output:
        latent_img_output.clear_output()
        display(image_to_use)

def generate_image_from_w(Gs, rnd_list, weight_list):
    device=torch.device('cuda')
    w_weight = [x/sum(weight_list) for x in weight_list]
    w_weight_list = []
    for i in range(len(rnd_list)):
        z_samples = torch.from_numpy(np.random.RandomState(rnd_list[i]).randn(1, Gs.z_dim)).to(device)
        w_samples = Gs.mapping(z_samples, None)
        w_samples_sq = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        w_samples_sq_list = list(np.squeeze(w_samples_sq))
        w_samples_sq_list_weight += w_weight[i] * w_samples_sq_list        
        
    w_opt = torch.tensor(w_samples_sq_list_weight, dtype=torch.float32, device=device, requires_grad=True)
    w_ws_image_to_use = (w_opt).repeat([1, Gs.mapping.num_ws, 1])
    w_synth_images = Gs.synthesis(ws_image_to_use, noise_mode='const')
        
    w_image_to_use = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#     image_to_use = Image.fromarray(image_to_use[0].cpu().numpy(), 'RGB').save('rand/' + str(rnd_seed[i]) + '.png')
    
    return w_image_to_use

def create_interactive_latent_controller():
    orig_img_output = widgets.Output()
    with orig_img_output:
        orig_img_output.clear_output()
        display(image_to_use)

    latent_img_output = widgets.Output()

    with latent_img_output:
        latent_img_output.clear_output()
        display(image_to_use)

    image_outputs = widgets.VBox([orig_img_output, latent_img_output])
    
    # collapse-hide
    generate_button = widgets.Button(description='Generate', layout=widgets.Layout(width='80%', height='15%', margin='20px'))
    generate_button.on_click(apply_latent_controls)

    reset_button = widgets.Button(description='Reset Controls', layout=widgets.Layout(width='80%', height='15%', margin='20px'))
    reset_button.on_click(reset_latent_controls)
    
    save_button = widgets.Button(description='Save Image', layout=widgets.Layout(width='80%', height='15%', margin='20px'))
    save_button.on_click(save_images)

    feature_sliders = []
    for feature in latent_controls:
        label = widgets.Label(feature)
        slider = widgets.FloatSlider(min=0, max=100)
        feature_sliders.append(widgets.HBox([slider, label]))
    feature_sliders.append(generate_button)
    feature_sliders.append(reset_button)
    feature_sliders.append(save_button)
    feature_sliders = widgets.VBox(feature_sliders)
    
    return widgets.HBox([image_outputs, feature_sliders], layout=Layout(margin='30px'))

device = torch.device('cuda')
################ Input - Start ####################
# path='C:/PythonWorkspace/stylegan2-ada-pytorch-main/out/chair/'
path = ''
pklfilename = 'chair_network-snapshot-003400.pkl'
csvfilename = 'test_output.csv'
results_size = 64
ref_size=64
adj = []
rndnumbers = []
score=[]
f = open(csvfilename, 'r', encoding='utf-8-sig')
rdr = csv.reader(f)
for line in rdr:
    rnd_id.append(line[0].rsplit('.')[0])
    adj.append(line[1])
    score.append(line[2])    
    
    
################ Input - End####################

# Gs, noise_vars, Gs_kwargs = load_model(pklfilename)
# # Latent space 평균 이미지 생성
# z_samples = np.random.RandomState(123).randn(1000, Gs.z_dim)
# w_samples = Gs.mapping(torch.from_numpy(z_samples).to(device), None)
# w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32) 
# w_avg = np.mean(w_samples, axis=0, keepdims=True)
# w_std = (np.sum((w_samples - w_avg) ** 2) / 10000) ** 0.5
# w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
pklfilename = 'chair_avg_file.pickle'
Gs, noise_vars, w_avg, ws_image_to_use, image_to_use = load_model(pklfilename)

# for step in range(1000):
#     # Learning rate schedule.
#     t = step / 1000
#     w_noise_scale = w_std * 0.05 * max(0.0, 1.0 - t / 0.75) ** 2

# w_noise = torch.randn_like(w_opt) * w_noise_scale
# ws_image_to_use = (w_opt + w_noise).repeat([1, Gs.mapping.num_ws, 1])
# synth_images = Gs.synthesis(ws_image_to_use, noise_mode='const')
# image_to_use = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
# image_to_use = Image.fromarray(image_to_use[0].cpu().numpy(), 'RGB')

targetName = 'projected_w.npz'

latent_code_list = []
latent_controls = []

for i in range(len(adj)):    
    adj_image_to_use = generate_image_from_w(Gs, rnd_list, weight_list)
    latent_code_list.append(adj_image_to_use)
#     latent_code_list.append(np.load('1. chair_image/' + adj[i] + '/' + targetName)['w'])
    latent_controls.append(adj[i])

controller = create_interactive_latent_controller()
controller
