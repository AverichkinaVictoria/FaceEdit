# FaceEdit
 
На рисунке продемонстрированы входное изображение и результат работы программы со следующими параметрами конфигурации:

•	Способ ввода – вектор атрибутов

•	Цвет волос – темный

•	Пол – женский

•	Возраст – молодой


```bash

python main.py --test_input_choice attrs --test_img_path test --test_img_name test.jpg --test_img_attrs 1 0 0 0 1

```



![Alt text](https://github.com/AverichkinaVictoria/FaceEdit/blob/dev/Screenshots/1.png)


 
На рисунке продемонстрированы входное изображение и результат работы программы со следующими параметрами конфигурации:

•	Способ ввода – текстовое описание

•	Текст-описание: «I wanna change my appearance. The hair should be blond, sex remains women, and i wanna see myself young.». 

```bash

python main.py --test_input_choice text --test_input_text 'I wanna change my appearance. The hair should be blond, sex remains women, and i wanna see myself young.' --test_img_path test --test_img_name test.jpg

```

![Alt text](https://github.com/AverichkinaVictoria/FaceEdit/blob/dev/Screenshots/2.png)



На рисунке продемонстрированы входное изображение и результат работы программы со следующими параметрами конфигурации:

•	Способ ввода – текстовое описание

•	Текст-описание: «Я хочу изменить свою внешность. Волосы перекрасить в коричневый, пол сменить на мужской, и увидеть себя в молодости.».

```bash
python main.py --test_input_choice text --test_input_text 'Я хочу изменить свою внешность. Волосы перекрасить в коричневый, пол сменить на мужской, и увидеть себя в молодости.' --test_img_path test --test_img_name test.jpg

```

![Alt text](https://github.com/AverichkinaVictoria/FaceEdit/blob/dev/Screenshots/3.png)

