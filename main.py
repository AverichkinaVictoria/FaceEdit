# *- coding: utf-8 -*-
import os
import argparse

from configurationObj import ConfigObj
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import telebot
from telebot import types
from pathlib import Path
from skimage import io

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    # if not os.path.exists(config.log_dir):
    #     os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()



# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # # Model configuration.
    # parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    # parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    # parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    # parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    # parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    # parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    # parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    # parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    # parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    # parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    # parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    # parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    #
    # # Training configuration.
    # parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    # parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    # parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    # parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    # parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    # parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    # parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    # parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    # parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    # parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
    #                     default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    #
    # # Test configuration.
    # parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    #
    # # Miscellaneous.
    # parser.add_argument('--num_workers', type=int, default=1)
    # parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    # parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    #
    # # Directories.
    # parser.add_argument('--celeba_image_dir', type=str, default='dataset/celeba/images')
    # parser.add_argument('--attr_path', type=str, default='dataset/celeba/list_attr_celeba.txt')
    # parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    # parser.add_argument('--log_dir', type=str, default='stargan/logs')
    # parser.add_argument('--model_save_dir', type=str, default='model_save')
    # parser.add_argument('--sample_dir', type=str, default='sample')
    # parser.add_argument('--result_dir', type=str, default='result')
    #
    # parser.add_argument('--test_input_choice', type=str, default='text', choices=['text', 'attrs'])
    # parser.add_argument('--test_input_text', type=str,
    #                     default='Я хочу изменить свою внешность. Волосы перекрасить в коричневый, пол сменить на мужской, и увидеть себя в молодости.')
    # # parser.add_argument('--test_input_text', type=str,default='I wanna change my appearance. The hair should be blond, sex remains women, and i wanna see myself young.')
    # parser.add_argument('--test_img_path', type=str, default='test')
    # parser.add_argument('--test_img_name', type=str, default='test.jpg')
    # parser.add_argument('--test_img_attrs', nargs='+', type=int, default=[1, 0, 0, 0, 1])
    #
    # # Step size.
    # parser.add_argument('--log_step', type=int, default=10)
    # parser.add_argument('--sample_step', type=int, default=1000)
    # parser.add_argument('--model_save_step', type=int, default=10000)
    # parser.add_argument('--lr_update_step', type=int, default=1000)


attrs = [1, 0, 0, 0, 1]
hair_color = 'black'
age = 'young'
sex = 'woman'
text = 'smth'
inp = 'attrs'
picture = [[]]
bot = telebot.TeleBot('5329656553:AAHIGQZ4JfvMjW_xbvC7oPkXj-JWZq7We5k')


@bot.message_handler(commands=["start"])
def start(m, res=False):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item5 = types.KeyboardButton("Описание")
    item3 = types.KeyboardButton("Список команд")
    markup.add(item3)
    markup.add(item5)
    bot.send_message(m.chat.id, 'Добро пожаловать в FaceEdit! Выберите интересующий раздел:',reply_markup=markup)

@bot.message_handler(commands=["input_choice_text"])
def input_choice_text(m, res=False):
    global attrs, hair_color, age, sex, text, inp, picture
    markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
    inp = 'text'
    msgText = bot.send_message(m.chat.id,
                               'Введите текст-описание результата редактирования изображения. Доступные параметры для изменения: светлый, темный и коричневый цвет волос; возраст - старый или молодой; пол - мужской или женский')
    bot.register_next_step_handler(msgText, step_Set_Text)


@bot.message_handler(commands=["load_image"])
def load_image(m, res=False):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    msgText = bot.send_message(m.chat.id,
                               'Загрузите изображение, содержащее лицо:')
    bot.register_next_step_handler(msgText, step_Set_Image)

@bot.message_handler(commands=["process"])
def process(m, res=False):
    global attrs, hair_color, age, sex, text, inp, picture
    print(picture)
    if (picture.size != 0):
        config = ConfigObj(inp, text, attrs, picture)
        print(config)
        main(config)
        photo = open('result/test-images.jpg', 'rb')
        bot.send_photo(m.chat.id, photo)
    else:
        bot.send_message(m.chat.id,'Загрузите изображение для обработки!')

@bot.message_handler(commands=["input_choice_attrs"])
def input_choice_attrs(m, res=False):
    global attrs, hair_color, age, sex, text, inp, picture
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("Темные волосы")
    item2 = types.KeyboardButton("Светлые волосы")
    item3 = types.KeyboardButton("Коричневые волосы")
    item4 = types.KeyboardButton("Женский пол")
    item5 = types.KeyboardButton("Мужской пол")
    item6 = types.KeyboardButton("Старый")
    item7 = types.KeyboardButton("Молодой")
    markup.add(item1)
    markup.add(item2)
    markup.add(item3)
    markup.add(item4)
    markup.add(item5)
    markup.add(item6)
    markup.add(item7)
    inp = 'attrs'
    bot.send_message(m.chat.id,'Выберите параметры результата редактирования изображения:', reply_markup=markup)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    # Если юзер прислал 1, выдаем ему случайный факт
    way = 'ss'
    global attrs, hair_color, age, sex, text, inp, picture
    if message.text.strip() == 'Описание' :
            way = 'Данный чат-бот предназначен для осуществления семантического редактирования лица. Для этого ' \
                  'необходимо загрузить изображения для редактирования. Затем выбрать способ ввода описания желаемого ' \
                  'результата редактирования изображения. Это может быть текст-описание результата или же выбор ' \
                  'параметров редактирования из предложенных. Для запуска процесса редактирования введите process '
    # Если юзер прислал 2, выдаем умную мысль
    elif message.text.strip() == 'Темные волосы':
        hair_color = 'black'
        attrs[0] = 1
        attrs[1] = 0
        attrs[2] = 0
        way = 'Текущие параметры обработанного изображения: \nЦвет волос: ' + hair_color + '\nПол: ' + sex + '\nВозраст: ' + age
    elif message.text.strip() == 'Светлые волосы':
        hair_color = 'blond'
        attrs[0] = 0
        attrs[1] = 1
        attrs[2] = 0
        way = 'Текущие параметры обработанного изображения: \nЦвет волос: ' + hair_color + '\nПол: ' + sex + '\nВозраст: ' + age
    elif message.text.strip() == 'Коричневые волосы':
        hair_color = 'brown'
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = 1
        way = 'Текущие параметры обработанного изображения: \nЦвет волос: ' + hair_color + '\nПол: ' + sex + '\nВозраст: ' + age
    elif message.text.strip() == 'Женский пол':
        sex = 'woman'
        attrs[3] = 0
        way = 'Текущие параметры обработанного изображения: \nЦвет волос: ' + hair_color + '\nПол: ' + sex + '\nВозраст: ' + age
    elif message.text.strip() == 'Мужской пол':
        sex = 'man'
        attrs[3] = 1
        way = 'Текущие параметры обработанного изображения: \nЦвет волос: ' + hair_color + '\nПол: ' + sex + '\nВозраст: ' + age

    elif message.text.strip() == 'Старый':
        age = 'old'
        attrs[4] = 0
        way = 'Текущие параметры обработанного изображения: \nЦвет волос: ' + hair_color + '\nПол: ' + sex + '\nВозраст: ' + age

    elif message.text.strip() == 'Молодой':
        age = 'young'
        attrs[4] = 1
        way = 'Текущие параметры обработанного изображения: \nЦвет волос: ' + hair_color + '\nПол: ' + sex + '\nВозраст: ' + age

    elif message.text.strip() == 'Список команд':
        way = '/input_choice_text – позволяет ввести текст-описание изменяемых параметров изображения \n/input_choice_attrs – позволяет выбрать список параметров результата обработки изображения \n/process – начинает процесс обработки изображения \n/load_image – загружает изображение для обработки'
    # Отсылаем юзеру сообщение в его чат
    bot.send_message(message.chat.id, way)

def step_Set_Text(message):
    global attrs, hair_color, age, sex, text, inp, picture
    cid = message.chat.id
    text = message.text
    print(text)


def step_Set_Image(message):
    cid = message.chat.id
    global attrs, hair_color, age, sex, text, inp, picture
    Path(f'files/{message.chat.id}/').mkdir(parents=True, exist_ok=True)
    if message.content_type == 'photo':
        raw = message.photo[1].file_id
        name = "test.jpg"
        file_info = bot.get_file(raw)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(name, 'wb') as new_file:
            new_file.write(downloaded_file)
        img = open(name, 'rb')
        image = io.imread(name)
        picture = image

@bot.message_handler(content_types=["photo"])
def echo_msg(message):
    global attrs, hair_color, age, sex, text, inp, picture
    Path(f'files/{message.chat.id}/').mkdir(parents=True, exist_ok=True)
    if message.content_type == 'photo':
        raw = message.photo[1].file_id
        name = "test.jpg"
        file_info = bot.get_file(raw)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(name, 'wb') as new_file:
            new_file.write(downloaded_file)
        img = open(name, 'rb')
        image = io.imread(name)
        picture = image

bot.polling(none_stop=True, interval=0)
