## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import random
import argparse
from cv2 import imwrite, imread
import os
import re

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_mnets import MNETSModel, ImagenetTF, NodeLookup
from preprocessing import preprocessing_factory

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

slim = tf.contrib.slim

def show(img_input, name=None):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img_input.flatten()+.5)*3
    if len(img) != 784 and name is not None:
        scaled = (0.5+img_input)*255
        imwrite(name, scaled)
        return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def prep_image(inp, height, width):
    image = tf.convert_to_tensor(inp)
    
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if inp.shape[0] != height or inp.shape[1] != width:
        # Resize the image to the specified height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
    image = tf.expand_dims(image, 0)
    image = tf.subtract(image, 0.5)

    #print(image.get_shape())
    
    return image

def generate_data(data_dir, input_classes, target_classes, height, width):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs_scaled = []
    
    input_image_names = []
    adv_image_names = []

    for filename in os.listdir(data_dir):
        if filename.startswith("input"):
            input_image_names.append(filename)
        elif filename.startswith("adv"):
            adv_image_names.append(filename)

    input_image_names.sort(key=lambda x: int(re.match("input(.*).png", x).group(1)))
    adv_image_names.sort(key=lambda x: int(re.match("adv(.*).png", x).group(1)))

    input_image_names = input_image_names
    adv_image_names = adv_image_names

    ret = []

    for (i, a) in zip(input_image_names, adv_image_names):
        img_num = int(re.match("input(.*).png", i).group(1))

        idx = int(img_num / len(input_classes))
        adv_idx = int(img_num % len(input_classes))
        
        input_c = input_classes[idx]
        target_c = target_classes[adv_idx]

        input_img_data = prep_image(imread(data_dir + i), height, width)
        adv_img_data = prep_image(imread(data_dir + a), height, width)

        ret.append([input_c, target_c, input_img_data, adv_img_data])
        
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checkpoint file')
    parser.add_argument('--ckpt', dest='ckpt', type=str,
                        help='Checkpoint file for model')
    parser.add_argument('--model', dest='model_name', type=str,
                        help='model_name')
    parser.add_argument('--norm', dest='norm', type=str,
                        help='distance metric')
    args = parser.parse_args()
    batch_size = 5

    with tf.Session() as sess:
        print("Running model {}".format(args.model_name))
        model = MNETSModel(args.ckpt, args.model_name, batch_size, sess)
        
        tf.train.start_queue_runners(sess)

        if args.norm == "0":
            norm = CarliniL0
        elif args.norm == "i":
            norm = CarliniLi
        else:
            norm = CarliniL2
   
        #input_classes = [455, 378, 98, 842, 595, 509, 888, 743, 362, 192]
        input_classes = [908, 532, 872, 476, 116, 836, 8, 144, 300, 751]
        target_classes = [893, 858, 350, 71, 948, 715, 558, 408, 349, 215]

        input_classes = input_classes
        target_classes = target_classes

        data = generate_data("./out_inception/out/images/images_c50/", input_classes, target_classes, model.image_size, model.image_size)

        X = tf.placeholder(tf.float32, shape=(1, model.image_size, model.image_size, 3))
        logits = model.predict(X)

        # attack = norm(sess, model, max_iterations=1000)

        if args.model_name == "inception_v3":
            variables_to_restore = slim.get_variables(scope="InceptionV3")
        elif args.model_name == "resnet_v2_152":
            variables_to_restore = slim.get_variables(scope="ResnetV2152")
        elif args.model_name.startswith("mobilenet"):
            variables_to_restore = slim.get_variables(scope="MobilenetV1")
        
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, args.ckpt)
        print("Checkpoint restored")
        
    	# Creates node ID --> English string lookup.
        node_lookup = NodeLookup()

        def topk(predictions):
            top_k = predictions.argsort()#[-FLAGS.num_top_predictions:][::-1]
            top_k = top_k[::-1]
            count = 1
            for node_id in top_k[:10]:
                #print('ID {}, score {}'.format(node_id, predictions[node_id]))
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                print("{}. {} (score = {})".format(count, human_string, score))
                count += 1

        def topk_labels(predictions):
            top_k = predictions.argsort()#[-FLAGS.num_top_predictions:][::-1]
            top_k = top_k[::-1]
            count = 1
            ret = []
            for node_id in top_k[:10]:
                score = predictions[node_id]
                ret.append((node_id, score))
            return ret

        num_transferred_targeted = 0
        num_transferred_untargeted = 0
        num_transferred_targeted_top_5 = 0
        pred_accuracy = 0
        pred_accuracy_top5 = 0
        for (input_c, target_c, input_img, adv_img) in data:
            pred = model.predict(input_img, reuse=True)
            pred = sess.run(pred)

            pred_adv = model.predict(adv_img, reuse=True)
            pred_adv = sess.run(pred_adv)

            pred = np.squeeze(pred)
            pred_adv = np.squeeze(pred_adv)

            topk_input = topk_labels(pred)
            topk_adv = topk_labels(pred_adv)

            print("Original class: {}, target class: {}".format(input_c, target_c))
            print("TopK pred for input: ", topk_input)
            print("TopK pred for adv: ", topk_adv)
            print("")
            
            if topk_adv[0][0] == target_c:
                num_transferred_targeted += 1
            elif topk_adv[0][0] != input_c:
                num_transferred_untargeted += 1
            if [topk_adv[i][0] == target_c for i in range(5)].count(True) > 0:
                num_transferred_targeted_top_5 += 1
                
            if topk_input[0][0] == input_c:
                pred_accuracy += 1
            if [topk_input[i][0] == input_c for i in range(5)].count(True) > 0:
                pred_accuracy_top5 += 1

        print("Prediction accuracy {} %".format(pred_accuracy * 100.0 / len(data)))
        print("Prediction accuracy, top 5 {} %".format(pred_accuracy_top5 * 100.0 / len(data)))
        print("Targeted transferability {} %".format(num_transferred_targeted * 100.0 / len(data)))
        print("Targeted transferability, top 5 {} %".format(num_transferred_targeted_top_5 * 100.0 / len(data)))
        print("Untargeted transferability {} %".format(num_transferred_untargeted * 100.0 / len(data)))
