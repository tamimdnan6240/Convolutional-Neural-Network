from pprint import pprint
import os
import setproctitle

class Config:
    name = 'DeepCrack_CT260_FT1'

    gpu_id = '0,1,2,3'

    setproctitle.setproctitle("%s" % name)

    # path
    train_data_path = 'data/train_path_index_proposed.txt' 
    test_data_path = "data/test_example_proposed.txt"  ## changed the name test_data_path from val data 
    checkpoint_path = './checkpoints' ## tamim: added ./ to save the pretrained model from current diretory to the checkpoints folder. 
    log_path = 'log' 
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 20  
    # visdom
    #vis_env = 'DeepCrack'  #Tamim: off visdom
    #port = 8097  #Tamim: off visdom
    #vis_train_loss_every = 40  #Tamim: off visdom
    #vis_train_acc_every = 40  #Tamim: off visdom
    #vis_train_img_every = 200 #Tamim: off visdom
    #val_every = 200    #Tamim: off visdom

    # training
    epoch = 100 ## Tamim: set epoch 100 instead 500
    pretrained_model = ''
    weight_decay = 0.0000
    lr_decay = 0.1
    lr = 1e-3 
    momentum = 0.9
    use_adam = True  # Use Adam optimizer
    train_batch_size = 16 ##tamim:  kept 16 for 200 * .8 = 160 training (10 sub steps)
    val_batch_size = 4 #Tamim: changed 4 for other 40 images (10 sub steps) 
    test_batch_size = 1 #

    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1

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
