import re
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
import numpy as np
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ArrowStyle
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans


def get_threshold(data):
    flattened_data = data.flatten()
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=0,n_init='auto').fit(flattened_data.reshape(-1, 1))
    cluster_labels = kmeans.labels_.reshape(data.shape)
    threshold = (kmeans.cluster_centers_[:, 0].min() + kmeans.cluster_centers_[:, 0].max()) / 2
    return threshold

def get_stable_epochs(acc_list,stable_patience=3):
    max_acc = acc_list[0]
    no_improvement_count = 0
    for i in range(2, len(acc_list)):
        if acc_list[i] <= max_acc:
            no_improvement_count += 1
            if no_improvement_count >= stable_patience:
                stable_epoch = i + 1  
                break
        else:
            max_acc = acc_list[i]
            no_improvement_count = 0
    return i
def count_layers(log_file):
    pattern_new_epoch = r'Test Acc:'
    pattern_dropped_acc = r'Rise Acc: (\d+\.\d+)'
    acc_sharp_dict = defaultdict(list) 
    loss_sharp_dict =  defaultdict(list)
    test_acc_list = [] 
    train_acc_list = [] 
    new_epoch = False
    layer_count = 0
    epoch_count = 0
    with open(log_file, 'r') as file:
        for line in file:
            if  'Test Acc:' in line and not 'Train Acc' in line and not 'SMax' in line:
                test_acc_pattern = re.compile(r'Acc:\s*([0-9]*\.?[0-9]+)')
                test_acc_match = test_acc_pattern.search(line)
                if test_acc_match:
                    test_acc_list.append(float(test_acc_match.group(1)))
                epoch_count += 1
            elif 'Origin' in line:
                train_acc_pattern = re.compile(r'Acc:\s*([0-9]*\.?[0-9]+)')
                train_acc_match = train_acc_pattern.search(line)
                if train_acc_match:
                    train_acc_list.append(float(train_acc_match.group(1)))
            else:
                if 'Dropped Acc:' in line and not 'Sharpness' in line:
                    layer_pattern = re.compile(r'INFO -\s*(.*?)\s*L:')
                    layer_match = layer_pattern.search(line)
                    if layer_match:
                        layer_name = layer_match.group(1)
                        if not layer_name[:-len(".weight")] in loss_sharp_dict.keys():
                            loss_sharp_dict[layer_name[:-len(".weight")]] = []
                            acc_sharp_dict[layer_name[:-len(".weight")]] = []
                if 'Sharpness_loss' in line:
                    layer_pattern = re.compile(r'INFO -\s*(.*?)\s*, Sharpness_loss:')
                    value_pattern = re.compile(r'Sharpness_loss:\s*([-+]?[0-9]*\.?[0-9]+)')
                    layer_match = layer_pattern.search(line)
                    value_match = value_pattern.search(line)
                    if layer_match:
                        layer_name = layer_match.group(1)
                        loss_sharp_dict[layer_name].append(float(value_match.group(1)))

                if 'Sharpness_acc' in line:
                    layer_pattern = re.compile(r'INFO -\s*(.*?)\s*, Sharpness_acc:')
                    value_pattern = re.compile(r'Sharpness_acc:\s*([-+]?[0-9]*\.?[0-9]+)')
                    
                    layer_match = layer_pattern.search(line)
                    value_match = value_pattern.search(line)
                    # print(line)
                    if layer_match:
                        layer_name = layer_match.group(1)
                        acc_sharp_dict[layer_name].append(float(value_match.group(1)))

    return loss_sharp_dict,acc_sharp_dict,test_acc_list,train_acc_list


# Do not forget put clean model first
label_list = ['clean','em','ops'] 
title_list = ['Vanilla','EM','OPS'] 


on_test = 'train' # 'train' 'test'
finetune = 'scratch' # 'scratch' 'finetune'
root_dir = '../exp/c10/r18'
suffix = 'defense=nodefense_lr=0.1_epochs=50_sharplr=0.1-0.1_iter=10-eps=0.05'

acc_data = []
loss_data = []
acc_dict_list = []
loss_dict_list = []
test_acc_list = [] 
train_acc_list = [] 
for idx,method in enumerate(label_list):
    log_file = os.path.join(root_dir,method,on_test,finetune,suffix,'output.log')
    
    loss_sharp_dict,acc_sharp_dict,test_acc,train_acc = count_layers(log_file)
    acc_data.append(np.array(list(acc_sharp_dict.values())).T)
    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)
    acc_dict_list.append(acc_sharp_dict)
    loss_dict_list.append(loss_sharp_dict)
    loss_data.append(np.array(list(loss_sharp_dict.values())).T) # 21,100 -> 100,21


threshold_epoch_list = []
for epoch in range(loss_data[0].T.shape[1]):
    threshold_epoch_list.append(get_threshold(loss_data[0].T[:,epoch]))
threshold = np.mean(threshold_epoch_list)
print('threshold:',threshold)


layer_count_list = []
stable_epochs_list = []
ud_list = []
acc_list = []
for idx,data in enumerate(loss_data):
    num_epochs = data.shape[0]
    layer_counts = np.sum(data > threshold) / num_epochs
    layer_count_list.append(layer_counts)
    acc_list.append(test_acc_list[idx][-1])

    ud = layer_counts / layer_count_list[0]
    ud_list.append(ud)
 
print(','.join(str(x) for x in layer_count_list))  
print(','.join(str(x) for x in ud_list))  