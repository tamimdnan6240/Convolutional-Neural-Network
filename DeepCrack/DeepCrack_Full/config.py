from pprint import pprint
import os
import setproctitle

class Config:
    name = 'DeepCrack_CT260_FT1'

    gpu_id = '0,1,2,3'

    setproctitle.setproctitle("%s" % name)

    # path
    train_data_path = 'data/train_path_index_original.txt'
    val_data_path = 'data/val_path_index.txt'
    checkpoint_path = 'checkpoints'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 20

    # visdom
    vis_env = 'DeepCrack'
    port = 8097
    vis_train_loss_every = 40
    vis_train_acc_every = 40
    vis_train_img_every = 120
    val_every = 200

    # training
    epoch = 200
    pretrained_model = ''
    weight_decay = 0.0005 # Tamim: changed to 0.0005
    lr_decay = 0.1
    lr = 1e-5 ## Tamim: according to paper
    momentum = 0.9
    use_adam = False  # Tamim: Use SGD accordint to paper, when you give false it will automatically take SGD. 
    train_batch_size = 16 ## Tamim: chnaged to 16
    val_batch_size = 16 ##CHanged to 16
    test_batch_size = 4

    acc_sigmoid_th = 0.5 
    pos_pixel_weight = 0.1

    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
