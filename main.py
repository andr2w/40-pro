import time 
import torch
import numpy as np
from importlib import import_module
import argparse 
import utils
import train 

parser = argparse.ArgumentParser(description='Fake-or-Real-News-Classification')
parser.add_argument('--model', type=str, default='BERTRNN', help= 'xxxx')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'data/'
    print(args.model)
    model_name = args.model
    x = import_module('model.' + model_name)
    config = x.Config(dataset)
    
    # make sure the result is the same 
    np.random.seed(1000)
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print('We are Loading the dataset now....')
    train_data, dev_data, test_data = utils.bulid_dataset(config)
    print('Successfully split the data into Train, dev, test')
    print('Now into the next, Build iterator')
    train_iter = utils.bulid_iterator(train_data, config)
    dev_iter = utils.bulid_iterator(dev_data, config)
    test_iter = utils.bulid_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print('Loading data cost: ', time_dif)

    # start training 
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)

