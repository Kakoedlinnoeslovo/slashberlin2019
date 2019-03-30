from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np
import random
from utils import *


# Set gpus
gpus = [0]
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])
num_gpus = len(gpus) # number of GPUs to use



### parameters setting
DATA_DIR = './datasets/CIHP'
LIST_PATH = './datasets/CIHP/list/train_rev.txt'
DATA_ID_LIST = './datasets/CIHP/list/train_id.txt'
SNAPSHOT_DIR = './checkpoint/CIHP_pgn'
LOG_DIR = './logs/CIHP_pgn'


N_CLASSES = 20
INPUT_SIZE = (512, 512)
BATCH_I = 1
BATCH_SIZE = BATCH_I * len(gpus)
SHUFFLE = True
RANDOM_SCALE = True
RANDOM_MIRROR = True
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
POWER = 0.9
p_Weight = 50
e_Weight = 0.005
Edge_Pos_W = 2
with open(DATA_ID_LIST, "r") as f:
    TRAIN_SET = len(f.readlines())
SAVE_PRED_EVERY = TRAIN_SET/BATCH_SIZE + 1 # save model per epoch (number of training set / batch)
NUM_STEPS = SAVE_PRED_EVERY * 100 + 1 #100 epoch

def main():
    RANDOM_SEED = random.randint(1000, 9999)
    tf.set_random_seed(RANDOM_SEED)

    coord = tf.train.Coordinator()

    with tf.name_scope("create_inputs"):
        reader = ImageReaderPGN(DATA_DIR, LIST_PATH, DATA_ID_LIST,
                                INPUT_SIZE, RANDOM_SCALE, RANDOM_MIRROR,
                                SHUFFLE, coord)
        image_batch, label_batch, edge_batch = reader.dequeue(BATCH_SIZE)

    tower_grads = []
    base_lr = tf.constant(LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.s

