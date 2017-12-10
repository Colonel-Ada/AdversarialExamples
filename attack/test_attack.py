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
from cv2 import imwrite

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_mnets import MNETSModel, ImagenetTF, NodeLookup

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


def generate_data(data, samples, num_targets=1, targeted=True, target_classes=None, start=0, imagenet=True):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    targets = []
    inputs_scaled = []

    print(data.test_data.shape)
    print(data.test_labels.shape)
    if not imagenet:
        seq = range(data.test_labels.shape[1])
    else:
        seq = target_classes if target_classes else random.sample(range(1,1001), num_targets)
    print("Target classes", seq)

    for i in range(samples):
        if targeted:
            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (imagenet == False):
                    continue
                inputs_scaled.append(data.test_data[start+i]/2.0)
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs_scaled.append(data.test_data[start+i]/2.0)
            targets.append(data.test_labels[start+i])

    inputs_scaled = np.array(inputs_scaled)
    #i = 1
    #for img in inputs_scaled:
    #    show(img, 'orig' + str(i) + '.png')
    #    i += 1    
    targets = np.array(targets)
    return inputs_scaled, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checkpoint file')
    parser.add_argument('--ckpt', dest='ckpt', type=str,
                        help='Checkpoint file for model')
    parser.add_argument('--model', dest='model_name', type=str,
                        help='model_name')
    parser.add_argument('--norm', dest='norm', type=str,
                        help='distance metric')
    parser.add_argument('--conf', dest='conf', type=int,
                       help='confidence')
    #parser.add_argument('--targets', dest='num_targets', type=int,
    #                    help='number of targets')
    args = parser.parse_args()
    batch_size = 5

    with tf.Session() as sess:
        print("Running model {}".format(args.model_name))
        model = MNETSModel(args.ckpt, args.model_name, batch_size, sess)
        data =  ImagenetTF(args.model_name, model.image_size) 
        
        tf.train.start_queue_runners(sess)
        data.get_batch(sess)

        if args.norm == "0":
            norm = CarliniL0
        elif args.norm == "i":
            norm = CarliniLi
        else:
            norm = CarliniL2
   
        target_classes = [893, 858, 350, 71, 948, 715, 558, 408, 349, 215]
        target_classes = target_classes
        attack = norm(sess, model, max_iterations=1000, confidence=args.conf)
        inputs, targets = generate_data(data, samples=len(target_classes),
                                        targeted=True, target_classes=target_classes,
                                        start=0, imagenet=True)

        print("Attack constructed")

        #print(tf.global_variables())
        if args.model_name == "inception_v3":
            variables_to_restore = slim.get_variables(scope="InceptionV3")
        elif args.model_name == "resnet_v2_152":
            variables_to_restore = slim.get_variables(scope="ResnetV2152")
        elif args.model_name.startswith("mobilenet"):
            variables_to_restore = slim.get_variables(scope="MobilenetV1")
        
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, args.ckpt)
        print("Checkpoint restored")
        
        print("Running attack")
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
        adv = adv.astype(np.float32)

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

        for i in range(len(adv)):
            print("Types:", inputs[i].dtype, adv[i].dtype)
            print("Valid:")
            show(inputs[i], name="input" + str(i) + ".png")
            print("Adversarial:")
            show(adv[i], name="adv" + str(i) + ".png")

            pred = model.predict(tf.convert_to_tensor(inputs[i:i+1]), reuse=True)
            pred_adv = model.predict(tf.convert_to_tensor(adv[i:i+1]), reuse=True)

            pred = sess.run(pred)
            pred_adv = sess.run(pred_adv)
           
            pred = np.squeeze(pred)
            pred_adv = np.squeeze(pred_adv)
 
            print("Original classification:")
            topk(pred)
            print("Adversarial classification:")
            topk(pred_adv)
            
            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
