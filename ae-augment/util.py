import numpy as np
import os
from PIL import Image

def make_partition(start,end,fraction):
    num = int(end*(1-fraction))
    partition = np.arange(start, end, dtype=np.int32)
    np.random.shuffle(partition)
    return partition[:num]

def locate(data_path, styles, max_label, partition):
    imageNameTrain, imageNameTest = [], []
    for style in styles:
        imageNameTrain.append([])
        imageNameTest.append([])
        path = os.path.join(data_path,'std/'+str(style)+'/cut')
        for i in range(max_label):
            fname = os.path.join(path,str(i)+'.png')
            if i not in partition:
                imageNameTrain[-1].append(fname)
            else:
                imageNameTest[-1].append(fname)
    return np.array(imageNameTrain), np.array(imageNameTest)

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

def numpy_append(batch,img):
    if batch is None: 
        batch = np.array([img], dtype=np.float32)
    else: 
        batch = np.concatenate((batch,np.array([img], dtype=np.float32)),axis=0)
    return batch

def img_loader(imgName,desired_height,desired_width,center_height,center_width,force_grayscale):
    image = Image.open(imgName)
    width, height = image.size
    img = Image.new('L',(desired_width,desired_height))
    background = np.ones_like(image,dtype=np.uint8)*200
    img.paste(Image.fromarray(background.astype(np.uint8)),(0,0))
    if width != center_width or height != center_height:
        image = image.resize((center_width, center_height), Image.BILINEAR)
    image_np = np.array(image).astype(np.uint8)
    #image_binary = np.greater(image_np,np.ones_like(image_np)*250).astype(np.uint8)
    img.paste(Image.fromarray(image_np),(int(desired_width/2)-int(center_width/2),int(desired_height/2)-int(center_height/2)))
    #img.paste(Image.fromarray(np.array(image).astype(np.uint8)),(0,0))
    image = img
    if force_grayscale: 
        image = image.convert("L")
    img = np.array(image)
    if len(img.shape) == 2: 
        img = img[:, :, None]
    #print(np.mean(np.power(img-255/2,2)-(255/2)*(255/2)))
    return img

def loader(imageName,idxes1,idxes2,charNum,desired_height,desired_width,value_range,augment,force_grayscale=True):
    length = idxes1.shape[0]
    fractions = [0.75, 0.5, 0.25]
    batch1, batch3, batch5, batch6 = None, None, None, None
    for i in range(length):
        styleId1 = int(idxes1[i] / charNum)
        charId1 = int(idxes1[i] % charNum)
        styleId2 = int(idxes2[i] / charNum)
        charId2 = int(idxes2[i] % charNum)
        batch1 = numpy_append(batch1,img_loader(imageName[styleId1,charId1],desired_height,desired_width,desired_height,desired_width,force_grayscale))
        batch3 = numpy_append(batch3,img_loader(imageName[styleId2,charId2],desired_height,desired_width,desired_height,desired_width,force_grayscale))
        batch5 = numpy_append(batch5,img_loader(imageName[styleId2,charId1],desired_height,desired_width,desired_height,desired_width,force_grayscale))
        batch6 = numpy_append(batch6,img_loader(imageName[styleId1,charId2],desired_height,desired_width,desired_height,desired_width,force_grayscale))
        for level in range(augment):
            batch1 = numpy_append(batch1,img_loader(imageName[styleId1,charId1],desired_height,desired_width,int(desired_height*fractions[level]),int(desired_width*fractions[level]),force_grayscale))
            batch3 = numpy_append(batch3,img_loader(imageName[styleId2,charId2],desired_height,desired_width,int(desired_height*fractions[level]),int(desired_width*fractions[level]),force_grayscale))
            batch5 = numpy_append(batch5,img_loader(imageName[styleId2,charId1],desired_height,desired_width,int(desired_height*fractions[level]),int(desired_width*fractions[level]),force_grayscale))
            batch6 = numpy_append(batch6,img_loader(imageName[styleId1,charId2],desired_height,desired_width,int(desired_height*fractions[level]),int(desired_width*fractions[level]),force_grayscale))
    #threshold = np.ones_like(batch1,dtype=np.float32)*1.0
    batch1 = (value_range[0] + (batch1 / 255.0) * (value_range[1] - value_range[0]))
    batch3 = (value_range[0] + (batch3 / 255.0) * (value_range[1] - value_range[0]))
    batch5 = (value_range[0] + (batch5 / 255.0) * (value_range[1] - value_range[0]))
    batch6 = (value_range[0] + (batch6 / 255.0) * (value_range[1] - value_range[0]))
    #batch1 = np.greater(batch1,threshold).astype(np.float32)
    #batch3 = np.greater(batch3,threshold).astype(np.float32)
    #batch5 = np.greater(batch5,threshold).astype(np.float32)
    #batch6 = np.greater(batch6,threshold).astype(np.float32)
    #print(np.mean(np.power(batch1-0.5,2)-0.25))
    #print(np.mean(np.power(batch3-0.5,2)-0.25))
    #print(np.mean(np.power(batch5-0.5,2)-0.25))
    #print(np.mean(np.power(batch6-0.5,2)-0.25))
    return batch1, batch3, batch5, batch6


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
