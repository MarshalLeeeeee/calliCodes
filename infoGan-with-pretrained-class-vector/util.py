import numpy as np
import os
from PIL import Image

def locate(data_path, styles=None, max_label=100):
    imageName, imageDict = [], {}
    if styles is None: styles = ['std-comp']
    for i in range(len(styles)):
        path = os.path.join(data_path,styles[i])
        for basepath, directories, fnames in os.walk(path):
            for fname in fnames:
                flabel = int(fname.split('/')[-1].split('.')[0].split('-')[0])
                suffix = fname.split('/')[-1].split('.')[-1]
                if suffix == 'png':
                    if flabel < max_label:
                        imageName.append(os.path.join(basepath,fname))
                        if flabel not in imageDict: imageDict[flabel] = []
                        imageDict[flabel].append(os.path.join(basepath,fname))
    return np.array(imageName), imageDict

def choice(image1,imageDict):
    image2 = []
    for fname in image1:
        flabel = int(fname.split('/')[-1].split('.')[0].split('-')[0])
        image2.append(np.random.choice(imageDict[flabel]))
    return image2

def find_truth(image1,imageTrue):
    image2 = []
    for fname in image1:
        flabel = int(fname.split('/')[-1].split('.')[0].split('-')[0])
        image2.append(imageTrue[flabel][0])
    return image2

def loader(imageName,desired_height,desired_width,value_range,force_grayscale=True):
    image_batch = None
    for fname in imageName:
        image = Image.open(fname)
        width, height = image.size
        if width != desired_width or height != desired_height:
            image = image.resize((desired_width, desired_height), Image.BILINEAR)
        if force_grayscale: 
            image = image.convert("L")
        img = np.array(image)
        if len(img.shape) == 2: 
            img = img[:, :, None]
        if image_batch is None: 
            image_batch = np.array([img], dtype=np.float32)
        else: 
            image_batch = np.concatenate((image_batch,np.array([img], dtype=np.float32)),axis=0)
    image_batch = (value_range[0] + (image_batch / 255.0) * (value_range[1] - value_range[0]))
    return image_batch

def plot(image_plot, image_reconstruct, epoch):
    num, w, h, c = image_plot.shape[0], image_plot.shape[1], image_plot.shape[2], image_plot.shape[3]
    img = Image.new('L',(w*2,h*num))
    for i in range(num):
        img.paste(Image.fromarray(np.squeeze((image_plot[i]*255).astype(np.uint8))),(64*0,64*i))
        img.paste(Image.fromarray(np.squeeze((image_reconstruct[i]*255).astype(np.uint8))),(64*1,64*i))
    img.save(os.path.join('savedImages',str(epoch)+'.png'))

def noise(batch_size,style_dim,class_vector):
    style_vector = np.random.standard_normal(size=(batch_size,style_dim))
    return np.concatenate([class_vector,style_vector],axis=1)

def get_vector(imageName,style_dim,continuous_dim):
    vector = None
    batch_size = imageName.shape[0]
    for name in imageName:
        v = np.load(name[:-3]+'npy')
        if vector is None: vector = np.array([v],dtype=np.float32)
        else: vector = np.concatenate((vector,np.array([v],dtype=np.float32)),axis=0)
    style_vector = np.random.standard_normal(size=(batch_size, style_dim))
    continuous_vector = np.random.uniform(-1.0, 1.0, size=(batch_size, continuous_dim))
    return np.concatenate((vector,style_vector,continuous_vector),axis=1)