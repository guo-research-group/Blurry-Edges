import argparse

def get_args(mode, big=False):
    parser = argparse.ArgumentParser()

    # basic settings
    parser.add_argument('--cuda', type=str, default='cuda:0', help='enable cuda')
    parser.add_argument('--model_path', type=str, default='./pretrained_weights', help='path of model weights')
    parser.add_argument('--img_size', type=int, default=(147,147), help='image size')
    parser.add_argument('--big_img_size', type=int, default=(587,587), help='big image size')
    parser.add_argument('--R', type=int, default=21, help='patch size')
    parser.add_argument('--w', type=float, default=1, help='w in loss function')
    parser.add_argument('--alpha_lambda', type=float, default=5e-3, help='rate for lambda in ridge regression')
    parser.add_argument('--cam_params', type=dict, default={'s': 0.1104, 'rho_1': 10.0, 'rho_2': 10.2, 'sigma_cam': 0.003, 'pixel_pitch': 5.86e-6}, help='camera parameters')
    parser.add_argument('--mag', type=float, default=4, help='magnification factor')

    # basic shape dataset generation
    if mode == 'data_gen_train_val':
        parser.add_argument('--data_path', type=str, default='./data/data_train_val', help='path of dataset')
        parser.add_argument('--num_sample_train', type=int, default=8000, help='number of samples to generate in training set')
        parser.add_argument('--num_sample_val', type=int, default=2000, help='number of samples to generate in validation set')
        parser.add_argument('--num_shape', type=tuple, default=(15,26), help='number of shapes to generate in each image')
        parser.add_argument('--Z_range', type=tuple, default=(0.75,1.18), help='distance between the object and the camera (m)')
        parser.add_argument('--alpha', type=tuple, default=(180,200), help='maximum average number of photons')
        parser.add_argument('--sigma', type=float, default=2, help='read noise coefficient')

    # local training
    elif mode == 'local_train':
        parser.add_argument('--data_path', type=str, default='./data/data_train_val/patches', help='path of dataset')
        parser.add_argument('--log_path', type=str, default='./logs', help='path of logs')
        parser.add_argument('--epoch_num', type=int, default=1000, help='number of epoches')
        parser.add_argument('--learning_rate', type=float, default=6e-5, help='initial learning rate for late training')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size') # 64
        parser.add_argument('--beta_bndry_loc', type=float, default=0.001, help='weight for boundary localization error')
        parser.add_argument('--beta_smthns', type=float, default=0.0005, help='weight for smoothness error')
        parser.add_argument('--dynamic_epoch', type=float, default=200, help='key epoch of dynamic scheduling')
    
    # global data pre-calculation
    elif mode == 'global_pre':
        parser.add_argument('--stride', type=int, default=2, help='stride')
        parser.add_argument('--data_path', type=str, default='./data/data_train_val', help='path of dataset')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    
    # global training
    elif mode == 'global_train':
        parser.add_argument('--stride', type=int, default=2, help='stride')
        parser.add_argument('--data_path', type=str, default='./data/data_train_val', help='path of dataset')
        parser.add_argument('--log_path', type=str, default='./logs', help='path of logs')
        parser.add_argument('--epoch_num', type=int, default=350, help='number of epoches')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate for late training')
        parser.add_argument('--batch_size', type=int, default=8, help='number of batch size')

        parser.add_argument('--gamma_color', type=tuple, default=(1.0,0.1,0.1), help='weight for color error')
        parser.add_argument('--gamma_color_cons', type=tuple, default=(0.2,0.1,0.05), help='weight for color consistency error')
        parser.add_argument('--gamma_bndry_cons', type=tuple, default=(0.05,0.05,0.02), help='weight for boundary consistency error')
        parser.add_argument('--gamma_smthns', type=tuple, default=(0.005,0.1,0.002), help='weight for smothness error')
        parser.add_argument('--gamma_smthns_cons', type=tuple, default=(0.005,0.1,0.002), help='weight for smothness consistency error')
        parser.add_argument('--gamma_bndry_loc', type=tuple, default=(0.0001,0.05,0.0001), help='weight for boundary localization error')
        parser.add_argument('--gamma_depth', type=tuple, default=(0.0001,0.05,0.5), help='weight for depth error')

        parser.add_argument('--dynamic_epoch', type=tuple, default=(30,100,200), help='key epoch of dynamic scheduling')
        parser.add_argument('--input_size', type=int, default=38, help='input layer size')
        parser.add_argument('--output_size', type=int, default=12, help='output layer size')

    # realistic dataset generation
    if mode == 'data_gen_test':
        parser.add_argument('--data_path', type=str, default='./data/data_test', help='path of dataset')
        parser.add_argument('--frgd_path', type=str, default='./data/MS_COCO_annotations/', help='path of MS COCO dataset')
        parser.add_argument('--bkgd_path', type=str, default='./data/Painting/', help='path of painting dataset')
        parser.add_argument('--num_sample_test', type=int, default=200, help='number of samples to generate in testing set')
        parser.add_argument('--Z_range', type=tuple, default=(0.75,1.18), help='distance between the object and the camera (m)')
        parser.add_argument('--alpha', type=tuple, default=(180,200), help='maximum average number of photons')
        parser.add_argument('--sigma', type=float, default=2, help='read noise coefficient')

    # evaluation
    elif mode == 'eval':
        parser.add_argument('--stride', type=int, default=2, help='stride')
        parser.add_argument('--log_path', type=str, default='./logs', help='path of logs')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--crop', type=int, default=10, help='center crop')
        parser.add_argument('--rho_prime', type=float, default=10.39, help='equivalent optical power for refocusing')
        parser.add_argument('--densify', type=str, default=None, help='densification method, set None to disable', choices=[None, 'pp', 'w'])
        if big:
            parser.add_argument('--n_margin_patch', type=int, default=10, help='number of margin patches that will be removed for global maps')
            parser.add_argument('--data_path', type=str, default='./data/data_test_big', help='path of dataset')
        else:
            parser.add_argument('--data_path', type=str, default='./data/data_test', help='path of dataset')

    return parser.parse_args()