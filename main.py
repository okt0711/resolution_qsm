import torch
from torch.utils.data import DataLoader
from os import makedirs
from os.path import join, isdir
from dataloader import QSMDataset
from options import Options
from models import ResolutionQSM

opt = Options().parse()
sdir = join(opt.save_path, opt.experiment_name)
opt.log_dir = join(sdir, 'log_dir')
opt.ckpt_dir = join(sdir, 'ckpt_dir')
opt.device = torch.device('cuda:' + str(opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
if not isdir(opt.ckpt_dir):
    makedirs(opt.ckpt_dir)

dataset_train = QSMDataset(opt, 'train')
dataset_test = QSMDataset(opt, 'test')

dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
dataloader_test.flist_P = dataset_test.flist

model = ResolutionQSM(opt)

if opt.training:
    model.train(dataloader_train)

model.test(dataloader_test)
