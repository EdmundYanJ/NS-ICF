import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dataset', '--dataset_folder', type=str, default='ml-1m', help='Set the data set path.')
parser.add_argument('-traind', '--train_dataset', type=str, default='train_data',
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument('-validd', '--valid_dataset', type=str, default='valid_data',
                    help='Set the data set for validing. All the data sets in the dataset folder are available.')
parser.add_argument('-testd', '--test_dataset', type=str, default='test_data',
                    help='Set the data set for testing. All the data sets in the dataset folder are available.')
parser.add_argument('-i', '--device_ids', type=str, default='0', help='Set the device (GPU ids). Split by @.'
                                                                       ' E.g., 0@2@3.')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('-e', '--epoch', type=int, default=41, help='Set the total epoch.')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Set the batch size.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Set the initial learning rate.')
parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.75, help='Set the learning rate decay rate.')
parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=10, help='Set the learning rate decay epoch.')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Set the weight decay (L2 penalty).')
parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th 5-fold validation, 0 <= ki < 5.')
parser.add_argument('-topk', '--topk_target', type=int, default=10, help='Do the i-th 5-fold validation, 0 < topk')
parser.add_argument('-rc', '--round_count', type=int, default=0, help='Count the round of experiments.')
parser.add_argument('-ma', '--master_address', type=str, default='166.111.5.60', help='Set the master address.')
parser.add_argument('-mp', '--master_port', type=str, default='33333', help='Set the master port.')
parser.add_argument('-li', '--log_iter', type=int, default=50, help='The number of iterations (batches) to log once.')

parser.add_argument('--use_not', action="store_true",
                    help='Use the NOT (~) operator in logical rules. '
                         'It will enhance model capability but make the RRL more complex.')
parser.add_argument('--save_best', action="store_true",
                    help='Save the model with best performance on the validation set.')
parser.add_argument('--estimated_grad', action="store_true",
                    help='Use estimated gradient.')
parser.add_argument('-s', '--structure', type=str, default='10@64',
                     help='Set the number of nodes in the binarization layer and logical layers. '
                          'E.g., 10@64, 10@64@32@16.')
parser.add_argument('-ts', '--top_structure', type=str, default='16',
                     help='Set the number of nodes in the top tower. '
                          'E.g., 10@64, 10@64@32@16.')
parser.add_argument('-div', '--div_place', type=str, default=None,
                     help='Set the features sep. '
                          'E.g., 0@10@20.')

rrl_args = parser.parse_args()
rrl_args.folder_name = '{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_topk{}_s{}_rc{}_useNOT{}_saveBest{}_estimatedGrad{}'.format(
    rrl_args.train_dataset, rrl_args.epoch, rrl_args.batch_size, rrl_args.learning_rate, rrl_args.lr_decay_rate,
    rrl_args.lr_decay_epoch, rrl_args.weight_decay, rrl_args.topk_target, rrl_args.structure, rrl_args.round_count, rrl_args.use_not,
    rrl_args.save_best, rrl_args.estimated_grad)

if not os.path.exists('log_folder'):
    os.mkdir('log_folder')
#rrl_args.folder_name = rrl_args.folder_name + '_L' + rrl_args.structure
rrl_args.set_folder_path = os.path.join('log_folder', rrl_args.dataset_folder)
if not os.path.exists(rrl_args.set_folder_path):
    os.mkdir(rrl_args.set_folder_path)
rrl_args.folder_path = os.path.join(rrl_args.set_folder_path, rrl_args.folder_name)
log_folder=os.path.join(rrl_args.set_folder_path, rrl_args.folder_name)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
rrl_args.model = os.path.join(log_folder, 'model.pth')
rrl_args.rrl_file = os.path.join(log_folder, 'rrl_origin.txt')
rrl_args.plot_file = os.path.join(log_folder, 'plot_file.pdf')
rrl_args.log = os.path.join(os.path.join(log_folder), 'log.txt')
#rrl_args.test_res = os.path.join(log_folder, 'test_res.txt')
rrl_args.test_res = os.path.join(log_folder, rrl_args.test_dataset+'.txt')
rrl_args.device_ids = list(map(int, rrl_args.device_ids.strip().split('@')))
rrl_args.gpus = len(rrl_args.device_ids)
rrl_args.nodes = 1
rrl_args.world_size = rrl_args.gpus * rrl_args.nodes
rrl_args.batch_size = int(rrl_args.batch_size / rrl_args.gpus)
