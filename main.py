import os
import datetime
import argparse
import shutil
import glob

from trainers.trainer import trainer_segmentation
from utils.utils import print_table

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='segmentation')
    parser.add_argument('--datatype', type=str, default='er')
    parser.add_argument('--onlineRandom', action='store_true', default=False)

    parser.add_argument('--train_data_list', type=str, default='er_train.txt')
    parser.add_argument('--predInvert', action='store_true', default=False) # if randomFlip > 50%, set True

    parser.add_argument('--is_iGTT', action='store_true', default=False)
    parser.add_argument('--use_evaluation', action='store_true', default=False)
    parser.add_argument('--num_window', type=int, default=30)
    parser.add_argument('--shift_r', type=int, default=1)
    parser.add_argument('--sample_p', type=int, default=0.1)
    parser.add_argument('--iGTT_dir', type=str, default='iGTT')

    parser.add_argument('--val_data_list', type=str, default='test.txt')
    parser.add_argument('--network', type=str, default='unet')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument("--segmentation_loss", type=str, default='bce')
    parser.add_argument("--estMask_loss", type=str, default='dmi')

    parser.add_argument("--optimizer_type", type=str, default='sgd')

    parser.add_argument("--normalization", action='store_true', default=True)

    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--in_img_size', type=int, default=256)
    parser.add_argument('--out_img_size', type=int, default=256)
    parser.add_argument('--init_lr', type=float, default=0.05)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_save_ckpt', type=int, default=80)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--print_steps', type=int, default=10)
    parser.add_argument('--save_every_x_epochs', type=int, default=2)
    parser.add_argument('--lr_decay_every_x_epochs', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--gpus', type=str, default="0,1,2,3")


    args = parser.parse_args()

    # parse train log dir
    train_log_dir = args.network + "_" + args.segmentation_loss + '_' +  args.optimizer_type
    args.train_data_list = args.train_data_list

    args.train_log_name = train_log_dir + "_" +  str(datetime.datetime.now().year) \
                    + str(datetime.datetime.now().month) \
                    + str(datetime.datetime.now().day)

    # if estimate mask, the save dir
    if args.datatype == 'er':
        args.iGTT_save_path = os.path.join('datasets/er/train', args.iGTT_dir, args.train_log_name)
        args.train_data_list = os.path.join('datasets/txt', 'er/train', args.train_data_list)
        args.val_data_list = os.path.join('datasets/txt', 'er', args.val_data_list)

    elif args.datatype == 'mito':
        args.iGTT_save_path = os.path.join('datasets/mito/train', args.iGTT_dir, args.train_log_name)
        args.train_data_list = os.path.join('datasets/txt', 'mito/train', args.train_data_list)
        args.val_data_list = os.path.join('datasets/txt', 'mito', args.val_data_list)

    elif args.datatype == "nuclei":
        args.iGTT_save_path = os.path.join('datasets/nuclei/train', args.iGTT_dir, args.train_log_name)
        args.train_data_list = os.path.join('datasets/txt', 'nuclei/train', args.train_data_list)
        args.val_data_list = os.path.join('datasets/txt', 'nuclei', args.val_data_list)

    if args.is_iGTT:
        args.train_log_dir = os.path.join('train_log', args.datatype, 'iGTT', args.train_log_name)
    else:
        args.train_log_dir = os.path.join('train_log', args.datatype, args.train_log_name)

    # parse checkpoints directory
    ckpt_dir = os.path.join(args.train_log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    args.ckpt_dir = ckpt_dir

    # parse log file text
    args.text_message_dir = args.train_log_dir + '/train_val_msg'
    os.makedirs(args.text_message_dir, exist_ok=True)

    args.log_file = os.path.join(args.text_message_dir, "log_file.txt")
    txt_list = glob.glob(os.path.join(args.text_message_dir, "*.txt"))
    if args.pretrain is None and os.path.exists(args.log_file):
        for file_path in txt_list:
            os.remove(file_path)

    # parse code backup directory
    code_backup_dir = os.path.join(args.train_log_dir, 'codes')
    os.makedirs(code_backup_dir, exist_ok=True)

    model_py = None
    if args.network == "unet":
        model_py = "unet.py"
        shutil.copy("./models/" + model_py, os.path.join(code_backup_dir, model_py))
    elif args.network == "deeplabv3+":
        model_py = "deeplab_v3.py"
        shutil.copy("./models/deeplabv3/" + model_py, os.path.join(code_backup_dir, model_py))

    # parse gpus
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpus
    gpu_list = []
    for str_id in args.gpus.split(','):
        id = int(str_id)
        gpu_list.append(id)
    args.gpu_list = gpu_list

    # format printing configs
    print("*" * 50)
    table_key = []
    table_value = []
    for key, value in vars(args).items():
        table_key.append(key)
        table_value.append(str(value))
    print_table([table_key, ["="] * len(vars(args).items()), table_value])

    # configure trainer and start training
    trainer = trainer_segmentation(args)
    trainer.train()

