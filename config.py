#coding:utf8
# 参数文件配置
class DefaultConfig():
    env = 'default' # visdom 环境
    model = 'alexnet' # 使用的模型，名字必须与models/__init__.py中的名字一致
    
    train_data_root = 'data/train/' # 训练集存放路径
    test_data_root = 'data/test/' # 测试集存放路径
    #load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    use_gpu = True # user GPU or not
    batch_size = 2 # batch size
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

#    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    pretrained = True
    max_epoch = 10
    lr = 0.001 # initial learning rate
    lr_decay = 0.9 # when val_loss increase, lr = lr*lr_decay
    step_size = 10 
    weight_decay = 1e-4 # 损失函数

    pretrained = True    
    train = True
#    test = False
    last_layer = 'classifier[6]' # alexnet
    last_layer_number = 1000
#    topk = 1
    
############### checkpoint恢复训练 ######################    
    resume = False
    checkpointfile = 'checkpoints/'
    resume_file = '' # 加载的模型
    start_epoch = 0
#########################################################
    tensorboard = True