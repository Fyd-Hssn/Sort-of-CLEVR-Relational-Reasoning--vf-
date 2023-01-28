"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
#Declaring and importing all required libraries
from __future__ import print_function
import argparse
import os
#import cPickle as pickle
import pickle
import random
import jax.numpy as jnp
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from model import RN, CNN_MLP


# Define all training settings such as Model Description, epochs, batch size, 
# learning rate etc.
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
# Adjusted default learning rate to 0.0002 from 0.0001 for marginal improvement in performance
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')

# method runs the parser and places the extracted data in a argparse.Namespace object
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available() # Returns a bool indicating if CUDA is currently available.

torch.manual_seed(args.seed) #Sets the seed for generating random numbers. Returns a torch.Generator object.
if args.cuda:
    torch.cuda.manual_seed(args.seed) # Sets the seed for generating random numbers for the current GPU

summary_writer = SummaryWriter() #Factory method to create a class wrapping a new SummaryWriter operation.

#Checking argument type for CNN or RN
if args.model=='CNN_MLP': 
  model = CNN_MLP(args) 
else:
  model = RN(args)
  
model_dirs = './model' #Defining the path for model.py
bs = args.batch_size #The batch size defines the number of samples that will be propagated through the network.
input_img = torch.FloatTensor(bs, 3, 75, 75) #A torch.Tensor is a multi-dimensional matrix containing elements of a float data type.
input_qst = torch.FloatTensor(bs, 18) #We define it for the imput img and input qst
label = torch.LongTensor(bs) #We use the samefunction as above to define label but use LONG instead

if args.cuda: #Checks if CUDA is present 
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img) #Once an object is assigned to a variable, you can refer to the object by that name. But the data itself is still contained within the object.
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(jnp.asarray(data[0][bs*i:bs*(i+1)])) #Creates a Tensor from a numpy.ndarray for img
    qst = torch.from_numpy(jnp.asarray(data[1][bs*i:bs*(i+1)])) #Creates a Tensor from a numpy.ndarray for qst
    ans = torch.from_numpy(jnp.asarray(data[2][bs*i:bs*(i+1)])) #Creates a Tensor from a numpy.ndarray for ans

    input_img.data.resize_(img.size()).copy_(img) #resize the image to desired size
    input_qst.data.resize_(qst.size()).copy_(qst) #resize qst to desired size
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data): #Check for current value table axis for tge image, qst and ans
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, ternary, rel, norel):
    model.train() #tells your model that you are training the model

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    #Reorganize the items present in the lists ternary, rel and norel
    random.shuffle(ternary) 
    random.shuffle(rel)
    random.shuffle(norel)

    #Passing these lists as argurment to the cvt_data_axis for conversion of axis
    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    #Declaring accuracy lists for ternary, rel and norel
    acc_ternary = []
    acc_rels = []
    acc_norels = []

    #Declaring list for loss of ternary, rel and norel 
    l_ternary = []
    l_binary = []
    l_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(ternary, batch_idx)
        accuracy_ternary, loss_ternary = model.train_(input_img, input_qst, label) #calculate loss and accuract for ternary data
        acc_ternary.append(accuracy_ternary.item()) #Appending the calculated accuracy
        l_ternary.append(loss_ternary.item()) #Appening the calculated loss

        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_img, input_qst, label) #calculate loss and accuract for relatioinal data
        acc_rels.append(accuracy_rel.item()) #Appending the calculated accuracy
        l_binary.append(loss_binary.item()) #Appening the calculated loss

        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_img, input_qst, label) #calculate loss and accuract for non-relational data
        acc_norels.append(accuracy_norel.item()) #Appending the calculated accuracy
        l_unary.append(loss_unary.item()) #Appening the calculated loss

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Ternary accuracy: {:.0f}% | Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_ternary,
                   accuracy_rel,
                   accuracy_norel))
    #calculating the average accuracy 
    avg_acc_ternary = sum(acc_ternary) / len(acc_ternary)
    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)

    summary_writer.add_scalars('Accuracy/train', {
        'ternary': avg_acc_ternary,
        'binary': avg_acc_binary,
        'unary': avg_acc_unary
    }, epoch)
    
    #calculating average loss
    avg_loss_ternary = sum(l_ternary) / len(l_ternary)
    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)

    summary_writer.add_scalars('Loss/train', {
        'ternary': avg_loss_ternary,
        'binary': avg_loss_binary,
        'unary': avg_loss_unary
    }, epoch)

    # return average accuracy
    return avg_acc_ternary, avg_acc_binary, avg_acc_unary

def test(epoch, ternary, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    #Passing these lists as argurment to the cvt_data_axis for conversion of axis
    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    #Declaring accuracy lists for ternary, rel and norel
    accuracy_ternary = []
    accuracy_rels = []
    accuracy_norels = []

    #Declaring list for loss of ternary, rel and norel 
    loss_ternary = []
    loss_binary = []
    loss_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(ternary, batch_idx)
        acc_ter, l_ter = model.test_(input_img, input_qst, label)
        accuracy_ternary.append(acc_ter.item())
        loss_ternary.append(l_ter.item())

        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())

    accuracy_ternary = sum(accuracy_ternary) / len(accuracy_ternary)
    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Ternary accuracy: {:.0f}% Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(
        accuracy_ternary, accuracy_rel, accuracy_norel))

    summary_writer.add_scalars('Accuracy/test', {
        'ternary': accuracy_ternary,
        'binary': accuracy_rel,
        'unary': accuracy_norel
    }, epoch)

    loss_ternary = sum(loss_ternary) / len(loss_ternary)
    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)

    summary_writer.add_scalars('Loss/test', {
        'ternary': loss_ternary,
        'binary': loss_binary,
        'unary': loss_unary
    }, epoch)

    return accuracy_ternary, accuracy_rel, accuracy_norel

    
def load_data():
    print('loading data...') 
    dirs = './data' #Setting path for data
    filename = os.path.join(dirs,'sort-of-clevr.pickle') #Finding CLEVR dataset
    with open(filename, 'rb') as f: #Opens file in binary format to read
      train_datasets, test_datasets = pickle.load(f) #loading train and test datasets
      
    # Declaring different lists to store the different train and test values
    ternary_train = []
    ternary_test = []
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')
    

    for img, ternary, relations, norelations in train_datasets:
        img = jnp.swapaxes(img, 0, 2) #Interchange two axes of an array
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_train.append((img,qst,ans)) #Appending the tenary training values into the list
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans)) #Appending the rel training values into the list
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans)) #Appending the norel training values into the list

    for img, ternary, relations, norelations in test_datasets:
        img = jnp.swapaxes(img, 0, 2)
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_test.append((img, qst, ans)) #Appending the tenary test values into the list
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans)) #Appending the rel test values into the list
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans)) #Appending the norel test values into the list
    
    return (ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test)
    

ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

with open(f'./{args.model}_{args.seed}_log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_acc_ternary', 'train_acc_rel',
                     'train_acc_norel', 'train_acc_ternary', 'test_acc_rel', 'test_acc_norel'])

    print(f"Training {args.model} {f'({args.relation_type})' if args.model == 'RN' else ''} model...")

    for epoch in range(1, args.epochs + 1):
        train_acc_ternary, train_acc_binary, train_acc_unary = train(
            epoch, ternary_train, rel_train, norel_train)
        test_acc_ternary, test_acc_binary, test_acc_unary = test(
            epoch, ternary_test, rel_test, norel_test)

        csv_writer.writerow([epoch, train_acc_ternary, train_acc_binary,
                         train_acc_unary, test_acc_ternary, test_acc_binary, test_acc_unary])
        model.save_model(epoch)
