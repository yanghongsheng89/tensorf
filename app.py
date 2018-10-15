# app.py
import tensorflow as tf
import model
import os
import numpy as np



from PIL import Image, ImageFilter

x = tf.placeholder('float',[None,784])
sess = tf.Session()
with tf.variable_scope("regression"):
    y,variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess,"data/regression")
with tf.variable_scope('converlution'):
    keep_prob = tf.placeholder('float')
    y2,variables = model.converlution(x,keep_prob)

saver = tf.train.Saver(variables)
saver.restore(sess,"data/converlution")

def regression(input):
    return sess.run(y,feed_dict={x:input}).flatten().tolist()
def converlution(input):
    return sess.run(y2,feed_dict={x:input,keep_prob:1.0}).flatten().tolist()


def imageprepare(path):
    
    im = Image.open(path).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1  
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    #newImage.save("sample.png")
    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva

if __name__ == '__main__':

    # img = Image.open('data/6.jpg').convert('L')
    # img.show()
    # k = list(img.getdata())
    k = imageprepare('data/xx.jpeg')
    # print(k)
    k = np.reshape(k,[1,784])
    # print(k)
    result = regression(k)
    # i = 0
    # for r in result:
    #     print(i,round(float(r*100),4))
    #     i = i + 1

    res = converlution(k)
    j = 0
    for r in res:
        print(j,round(float(r*100),4),'----',round(float(result[j]* 100),4))
        j = j + 1
