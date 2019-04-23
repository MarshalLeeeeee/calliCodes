import numpy as np
import os
from PIL import Image

def locate(data_path, styles=None, max_label=100, partition=[]):
    imageNameTrain, imageDictTrain = [], {}
    imageNameTest, imageDictTest = [], {}
    if styles is None: styles = ['std-comp']
    for i in range(len(styles)):
        path = os.path.join(data_path,styles[i])
        for basepath, directories, fnames in os.walk(path):
            cnt = 0
            for fname in fnames:
                flabel = int(fname.split('/')[-1].split('.')[0].split('-')[0])
                suffix = fname.split('/')[-1].split('.')[-1]
                if suffix == 'png':
                    if flabel < max_label:
                        if flabel in partition:
                            imageNameTest.append(os.path.join(basepath,fname))
                            if flabel not in imageDictTest: imageDictTest[flabel] = []
                            imageDictTest[flabel].append(os.path.join(basepath,fname))
                        else:
                            imageNameTrain.append(os.path.join(basepath,fname))
                            if flabel not in imageDictTrain: imageDictTrain[flabel] = []
                            imageDictTrain[flabel].append(os.path.join(basepath,fname))
                cnt += 1
    return np.array(imageNameTrain), imageDictTrain, np.array(imageNameTest), imageDictTest

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

def plot(image1_plot, image2_plot, image1_reconstruct, image2_reconstruct, title, epoch, reconstruct_coef_1, reconstruct_coef_2, lr):
    num, w, h, c = image1_plot.shape[0], image1_plot.shape[1], image1_plot.shape[2], image1_plot.shape[3]
    img = Image.new('L',(w*4,h*num))
    for i in range(num):
        img.paste(Image.fromarray(np.squeeze((image1_plot[i]*255).astype(np.uint8))),(64*0,64*i))
        img.paste(Image.fromarray(np.squeeze((image2_plot[i]*255).astype(np.uint8))),(64*1,64*i))
        img.paste(Image.fromarray(np.squeeze((image1_reconstruct[i]*255).astype(np.uint8))),(64*2,64*i))
        img.paste(Image.fromarray(np.squeeze((image2_reconstruct[i]*255).astype(np.uint8))),(64*3,64*i))
    img.save(os.path.join('savedImages',title+'-'+str(epoch)+'-'+str(reconstruct_coef_1)+'-'+str(reconstruct_coef_2)+str(lr)+'-''.png'))

def plot_batch(images, title, epoch, coefs):
    length = len(images)
    num, w, h, c = images[0].shape[0], images[0].shape[1], images[0].shape[2], images[0].shape[3]
    img = Image.new('L',(w*length,h*num))
    for i in range(num):
        for j in range(length):
            img.paste(Image.fromarray(np.squeeze((images[j][i]*255).astype(np.uint8))),(64*j,64*i))
    imgName = title+'-'+str(epoch)
    for coef in coefs:
        imgName += '-'+str(coef)
    img.save(os.path.join('savedImages',imgName+'.png'))

def save_vector(imageName, vector):
    path = '../../demo'
    for i in range(len(imageName)):
        namesp = imageName[i].split('/')
        style = namesp[-2]
        filename = namesp[-1].split('.')[0]
        np.save(os.path.join(os.path.join(path,style),filename+'.npy'),vector[i])
