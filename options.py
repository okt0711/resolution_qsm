import argparse
import os


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--act_type', type=str, default='lrelu', help='type of activation function')
        self.parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='moment for optimizer, 0 to 1')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='moment for optimizer, 0 to 1')
        self.parser.add_argument('--continue_epoch', type=int, default=0, help='continue training from continue_epoch')
        self.parser.add_argument('--data_root', type=str, default='', help='path for data')
        self.parser.add_argument('--disp_step', type=int, default=100, help='display step')
        self.parser.add_argument('--experiment_name', type=str, default='experiment_name', help='experiment name')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu_ids')
        self.parser.add_argument('--init_gain', type=float, default=1, help='gain of weight initialization')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='the type of weight initialization method')
        self.parser.add_argument('--lambda_cycle', type=float, default=1, help='lambda for cycle-consistency loss')
        self.parser.add_argument('--lambda_grad', type=float, default=0.05, help='lambda for gradient loss')
        self.parser.add_argument('--lambda_tv', type=float, default=0.01, help='lambda for tv loss')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--lr_decay_iters', type=int, default=0, help='lr decay iteration')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='the type of lr decay')
        self.parser.add_argument('--nX', type=int, default=64, help='width of images')
        self.parser.add_argument('--nY', type=int, default=64, help='height of images')
        self.parser.add_argument('--nZ', type=int, default=64, help='depth of images')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='the number of epoch iteration')
        self.parser.add_argument('--n_epochs_decay', type=int, default=0, help='the number of decay epoch iteration')
        self.parser.add_argument('--ngf', type=int, default=32, help='the number of network filters')
        self.parser.add_argument('--num_patch', type=int, default=5000, help='number of random patch')
        self.parser.add_argument('--save_epoch', type=int, default=10, help='save model per N epochs')
        self.parser.add_argument('--save_path', type=str, default='', help='path for saving results')
        self.parser.add_argument('--stride', type=int, default=32, help='stride')
        self.parser.add_argument('--test_epoch', type=int, default=100, help='test using the model of test_epoch')
        self.parser.add_argument('--training', action='store_true', help='True if training mode')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('---------- Options ----------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('------------ End ------------')

        expr_dir = os.path.join(self.opt.save_path, self.opt.experiment_name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('---------- Options ----------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('------------ End ------------')
        opt_file.close()
        return self.opt

