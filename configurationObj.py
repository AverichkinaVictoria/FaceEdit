class ConfigObj(object):

    def __init__(self, test_input_choice,test_input_text,test_img_attrs, image):
      self.c_dim = 5
      self.celeba_crop_size = 178
      self.image_size = 256
      self.g_conv_dim = 64
      self.d_conv_dim = 64
      self.g_repeat_num = 6
      self.d_repeat_num = 6
      self.lambda_cls = 1
      self.lambda_rec = 10
      self.lambda_gp = 10
      self.dataset = 'CelebA'
      self.batch_size = 16
      self.num_iters = 200000
      self.num_iters_decay = 100000
      self.g_lr = 0.0001
      self.d_lr = 0.0001
      self.n_critic = 5
      self.beta1 = 0.5
      self.beta2 = 0.999
      self.resume_iters = None
      self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
      self.test_iters = 200000
      self.num_workers = 1
      self.mode = 'test'
      self.celeba_image_dir = 'dataset/celeba/images'
      self.attr_path = 'dataset/celeba/list_attr_celeba.txt'
      self.model_save_dir = 'model_save'
      self.sample_dir = 'sample'
      self.result_dir = 'result'
      self.test_input_choice = test_input_choice
      self.test_input_text = test_input_text
      self.test_img_path = 'test'
      self.test_img_name = image
      self.test_img_attrs = test_img_attrs

      self.log_step = 10
      self.sample_step = 1000
      self.model_save_step = 10000
      self.lr_update_step = 1000
