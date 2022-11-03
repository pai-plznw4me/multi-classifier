import math
import multiprocessing
from copy import copy

import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os
from tensorflow.keras.callbacks import LearningRateScheduler


class TrainClassifier:
    """
    Description:
        Keras Generator Dataprovider을 입력으로 받아 모델을 학습 할 수 있는 모델 입니다.
        TrainClassifier 생성자에서는 backbone, header, learning_rate, optimizer  등
        모델을 학습하기 위한 설정을 지정합니다. 설정이 지정되면 TrainClassifier.training() 으로 지정된 모델을 학습 할 수 있습니다.
        학습 된 모델은 TrainClassifier.model_dir 에 저장되고 학습 로그는 TrainClassifier.log_dir 에 저장됩니다.
    :key float learning_rate:
    :key str log_dir: 학습 로그가 저장될 폴더 example)
    :key str model_dir: 학습 된 모델이 저장될 폴더
    :key str optimizer_name: optimizer 이름 (adam ,sgd, rmsprop, momentum)
    :key bool preproc: backbone 모델의 input image proprecessing algorithm 적용 여부
    :key str backbone_name: backbone model 이름
    :key bool pretrain: pretrained weights 사용 여부 결정
    :key bool freezing: backbone 을 freezing 여부 결정, True 이면 backbone weights 을 변경하지 않음
    :key int n_epochs: 총 1 epochs 회 수
    :key int step_per_epoch: 1 epcoh 당 진행될 step 수
    :key int n_classes: 출력 개 수

    Usage 1: 전체 공개 등록번호 학습 및 전체 공개 등록번호 평가 데이터로 평가
        train_classifier = TrainClassifier(**kwargs)

    """

    def __init__(self, **hparam):
        # 외부 Agument 을 인자로 받습니다.
        self.lr = hparam['lr']
        self.log_dir = hparam['log_dir']
        self.model_dir = hparam['model_dir']
        self.opt = hparam['opt']
        self.preproc = hparam['preproc']
        self.n_epochs = hparam['n_epochs']
        self.step_per_epoch = hparam['step_per_epoch']
        self.backbone_name = hparam['backbone_name']
        self.pretrain = hparam['pretrain']
        self.freezing = hparam['freezing']
        self.n_classes = int(hparam['n_classes'])

        self.preprocess_input = None
        self.model = None
        self.batch_size = None
        self.lr_scheduler = None
        self.n_worker = None
        self.queue_size = None
        self.input_shape = None
        self.weights = None
        self.backbone = None
        self.log_writer = None
        self.loss = None
        self.metrics = None

    def set_model(self, input_shape):
        """

        Args:
        @key input_shape: (h, w, ch)

        Returns:

        """
        # set input
        self.set_input(input_shape)

        # set backbone
        self.set_pretrain()
        self.set_backbone()
        self.set_freezing()

        # set header
        self.set_header()

    def set_hparam(self):
        # model compile 하기 위해 optimzier, loss , metrics 설정
        self.set_optimizer()
        self.set_loss()
        self.set_metrics()

        # keras model 생성
        self.set_compile()

        # learning rate scheduler 을 지정합니다. 만약 지정하지 않으면 위 set optimizer 에서 지정한 고정 learning rate 을 사용합니다.
        self.set_dynamic_lr()
        self.set_multiprocessing()

    def set_input(self, input_shape=(None, None, 3)):
        """
        Description:
            tensorflow keras model 에 입력될 input 을 결정합니다.
        Args:
            :param tuple input_shape:
        Returns:
        """
        self.input_shape = input_shape

    def set_pretrain(self):
        """
        Description:
            pretrained 된 weight 의 사용 여부를 check 합니다.
            self.pretrain 이 True 이면 imagenet 데이터를 사용하빈다.
            self.pretrain 이 False 이면 weights 을 초기화 해 사용합니다.
        Returns:
        """
        if self.pretrain:
            self.weights = "imagenet"
            print('ImageNet Pretrain 가 적용되었습니다.')
        else:
            self.weights = None
            print('Weight 을 초기화 해 사용합니다.')

    def set_preprocess_input(self, backbone):
        """
        Description:
            backbone 인자로 받는 모델명에 따라 preprocess_input 을 불러와주는 함수입니다.
        :param backbone: string, config 파일에서 불러온 모델명이 들어있습니다.
        :return:
            preprocess_input: 불러온 백본 모델의 preprocess_input 를 리턴합니다.
        """
        if backbone == "xception":
            from tensorflow.keras.applications.xception import preprocess_input

        elif backbone == "vgg16":
            from tensorflow.keras.applications.vgg16 import preprocess_input

        elif backbone == "vgg19":
            from tensorflow.keras.applications.vgg19 import preprocess_input

        elif backbone == "resnet50" or backbone == "resnet101" or backbone == "resnet152":
            from tensorflow.keras.applications.resnet import preprocess_input

        elif backbone == "resnet50v2" or backbone == "resnet101v2" or backbone == "resnet152v2":
            from tensorflow.keras.applications.resnet_v2 import preprocess_input

        elif backbone == "inceptionv3":
            from tensorflow.keras.applications.inception_v3 import preprocess_input

        elif backbone == "inceptionresnetv2":
            from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

        elif backbone == "mobilenet":
            from tensorflow.keras.applications.mobilenet import preprocess_input

        elif backbone == "mobilenetv2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        elif backbone == "densenet121" or backbone == "densenet169" or backbone == "densenet201":
            from tensorflow.keras.applications.densenet import preprocess_input

        elif backbone == "efficientnetb0" or backbone == "efficientnetb1" or backbone == "efficientnetb2" or \
                backbone == "efficientnetb3" or backbone == "efficientnetb4" or backbone == "efficientnetb5" or \
                backbone == "efficientnetb6" or backbone == "efficientnetb7":
            from tensorflow.keras.applications.efficientnet import preprocess_input

        else:
            assert True, print("지원하지 않는 모델입니다.")

        return preprocess_input

    def set_backbone(self):
        """
        Description:
            백본 모델을 설정합니다.
            아래 리스트에 해당하는 모델을 backbone 으로 지정 할 수 있습니다.
            해당 모델은 기본적으로 imagenet pretrined weights 을 사용합니다.
            self.set_pretrain() 에서 weight 초기화를 결정할수 있습니다.
                -xception
                -vgg16
                -vgg19
                -resnet50
                -resnet101
                -resnet152
                -resnet50v2
                -resnet101v2
                -resnet152v2
                -inceptionv3
                -inceptionresnetv2
                -mobilenet
                -mobilenetv2
                -densenet121
                -densenet169
                -densenet201
                -efficientnetb0
                -efficientnetb1
                -efficientnetb2
                -efficientnetb3
                -efficientnetb4
                -efficientnetb5
                -efficientnetb6
                -efficientnetb7
        :return:
            model: 백본 모델을 리턴합니다.
        """
        print('Backbone :  {}'.format(self.backbone_name))
        if self.backbone_name == "xception":
            self.backbone = tf.keras.applications.Xception(include_top=False, weights=self.weights,
                                                           input_shape=self.input_shape)

        elif self.backbone_name == "vgg16":
            self.backbone = tf.keras.applications.VGG16(include_top=False, weights=self.weights,
                                                        input_shape=self.input_shape)

        elif self.backbone_name == "vgg19":
            self.backbone = tf.keras.applications.VGG19(include_top=False, weights=self.weights,
                                                        input_shape=self.input_shape)

        elif self.backbone_name == "resnet50":
            self.backbone = tf.keras.applications.ResNet50(include_top=False, weights=self.weights,
                                                           input_shape=self.input_shape)

        elif self.backbone_name == "resnet101":
            self.backbone = tf.keras.applications.ResNet101(include_top=False, weights=self.weights,
                                                            input_shape=self.input_shape)

        elif self.backbone_name == "resnet152":
            self.backbone = tf.keras.applications.ResNet152(include_top=False, weights=self.weights,
                                                            input_shape=self.input_shape)

        elif self.backbone_name == "resnet50v2":
            self.backbone = tf.keras.applications.ResNet50V2(include_top=False, weights=self.weights,
                                                             input_shape=self.input_shape)

        elif self.backbone_name == "resnet101v2":
            self.backbone = tf.keras.applications.ResNet101V2(include_top=False, weights=self.weights,
                                                              input_shape=self.input_shape)

        elif self.backbone_name == "resnet152v2":
            self.backbone = tf.keras.applications.ResNet152V2(include_top=False, weights=self.weights,
                                                              input_shape=self.input_shape)

        elif self.backbone_name == "inceptionv3":
            self.backbone = tf.keras.applications.InceptionV3(include_top=False, weights=self.weights,
                                                              input_shape=self.input_shape)

        elif self.backbone_name == "inceptionresnetv2":
            self.backbone = tf.keras.applications.InceptionResNetV2(include_top=False, weights=self.weights,
                                                                    input_shape=self.input_shape)

        elif self.backbone_name == "mobilenet":
            self.backbone = tf.keras.applications.MobileNet(include_top=False, weights=self.weights,
                                                            input_shape=self.input_shape)

        elif self.backbone_name == "mobilenetv2":
            self.backbone = tf.keras.applications.MobileNetV2(include_top=False, weights=self.weights,
                                                              input_shape=self.input_shape)

        elif self.backbone_name == "densenet121":
            self.backbone = tf.keras.applications.DenseNet121(include_top=False, weights=self.weights,
                                                              input_shape=self.input_shape)

        elif self.backbone_name == "densenet169":
            self.backbone = tf.keras.applications.DenseNet169(include_top=False, weights=self.weights,
                                                              input_shape=self.input_shape)

        elif self.backbone_name == "densenet201":
            self.backbone = tf.keras.applications.DenseNet201(include_top=False, weights=self.weights,
                                                              input_shape=self.input_shape)

        elif self.backbone_name == "efficientnetb0":
            self.backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights=self.weights,
                                                                 input_shape=self.input_shape)

        elif self.backbone_name == "efficientnetb1":
            self.backbone = tf.keras.applications.EfficientNetB1(include_top=False, weights=self.weights,
                                                                 input_shape=self.input_shape)

        elif self.backbone_name == "efficientnetb2":
            self.backbone = tf.keras.applications.EfficientNetB2(include_top=False, weights=self.weights,
                                                                 input_shape=self.input_shape)

        elif self.backbone_name == "efficientnetb3":
            self.backbone = tf.keras.applications.EfficientNetB3(include_top=False, weights=self.weights,
                                                                 input_shape=self.input_shape)

        elif self.backbone_name == "efficientnetb4":
            self.backbone = tf.keras.applications.EfficientNetB4(include_top=False, weights=self.weights,
                                                                 input_shape=self.input_shape)

        elif self.backbone_name == "efficientnetb5":
            self.backbone = tf.keras.applications.EfficientNetB5(include_top=False, weights=self.weights,
                                                                 input_shape=self.input_shape)

        elif self.backbone_name == "efficientnetb6":
            self.backbone = tf.keras.applications.EfficientNetB6(include_top=False, weights=self.weights,
                                                                 input_shape=self.input_shape)

        elif self.backbone_name == "efficientnetb7":
            self.backbone = tf.keras.applications.EfficientNetB7(include_top=False, weights=self.weights,
                                                                 input_shape=self.input_shape)
        else:
            raise NotImplementedError
            assert True, print("지원하지 않는 모델입니다.")

    def set_freezing(self):
        """
        Description:
            weight 을 freezing 할지 아닐지를 결정합니다.
            True 이면 backbone 을 학습 시키지 않습니다.
            False 이면 backbone 을 학습 합니다.
        Returns:
        """
        if self.freezing:
            self.backbone.trainable = False
            print('backbone weights 을 고정합니다.')
        else:
            self.backbone.trainable = True

    def set_preproc(self):
        """
        preprocessing 을 적용합니다.
        @return:
        """
        if self.preproc:
            self.preprocess_input = self.set_preprocess_input(self.backbone_name)
            print('{} Preprocessing 을 적용합니다.'.format(self.backbone_name))
        else:
            print('{} Preprocessing 을 적용하지 않습니다.'.format(self.backbone_name))

    def set_header(self):
        """
        Description:
            모델의 출력 부를 결정합니다.
            해당 function 을 override 해 사용합니다.
        Returns:
        """
        # head
        layer = GlobalAveragePooling2D()(self.backbone.output)

        # header
        layer = Dense(128)(layer)
        layer = Dropout(rate=0.5)(layer)
        layer = ReLU()(layer)
        layer = Dense(128)(layer)
        layer = Dropout(rate=0.5)(layer)
        layer = ReLU()(layer)
        layer = Dense(self.n_classes, activation='softmax')(layer)
        self.model = Model(self.backbone.input, layer)

    def set_optimizer(self):
        """
        Description:
        keras optimizer 을 결정합니다.
         - adam, sgd, rmsprop, momumtum(with nesterov)
        :return: tensorflow.keras.optimizer
        """
        if self.opt == 'adam':
            self.opt = Adam(lr=self.lr)

        elif self.opt == 'sgd':
            self.opt = SGD(lr=self.lr)

        elif self.opt == 'rmsprop':
            self.opt = RMSprop(lr=self.lr)

        elif self.opt == 'momentum':
            self.opt = SGD(lr=self.lr, momentum=0.9, nesterov=True)

        else:
            raise ValueError
        print("Optimizer : {}".format(type(self.opt).__name__))

    def generate_folder(self):
        """
        Description:
            지정된 폴더에 deep learning model 을 저장하는 폴더를 생성합니다.
            지정된 폴더에 deep learning model 학습 로그를 저장하는 폴더를 생성합니다.
            폴더 생성 원리는 이미 생성된 폴더가 있으면 숫지를 1 씩 증가해 생성합니다.
             ./models/0 -> ./models/0_1 -> ./models/0_2
             ./logs/0 -> ./logs/0_1 -> ./logs/0_2
        Returns:
        """
        # generate model
        self.model_dir = generate_tmp_folder(self.model_dir)
        print("model file 이 저장될 장소 : {}".format(self.model_dir))
        self.log_dir = generate_tmp_folder(self.log_dir)
        print("log file 이 저장될 장소 : {}".format(self.log_dir))

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 기록될 로그
        log_path = os.path.join(self.log_dir, "logs.txt")
        self.log_writer = open(log_path, 'w')
        self.log_writer.write("{}\t{}\t{}\t{}\t{}\n".format("train_loss", "train_acc", "val_loss", "val_acc", "lr"))

    def set_loss(self):
        self.loss = 'categorical_crossentropy'

    def set_metrics(self):
        self.metrics = ['acc']

    def set_compile(self):
        """
        Description:
         loss, optimizer, loss 을 입력으로 keras model 을 생성합니다.
        Returns:
        """

        self.model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics)

    def set_dynamic_lr(self):
        """
        Description:
        동적 learning rate 을 구현하기 위해 keras callback learning rate function 을 생성합니다.
        Returns:
        """

        # Learning rate schedule
        def step_decay(epoch):
            initial_lrate = self.lr
            drop = 0.0
            epochs_drop = 20.0
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        self.lr_scheduler = LearningRateScheduler(step_decay)

    def set_multiprocessing(self):
        """
        Description:
            Multi processing 시작시 해당 함수를 시작해야 합니다.
            self.training() 함수 이전에 실행해야 합니다.
        Usage:
            self.set_multiprocessing()
            self.training()
        @return:
        """
        self.n_worker = multiprocessing.cpu_count() * 2 + 1
        self.queue_size = 64 * (self.n_worker * 2)

        print('multi processing 을 활용합니다. \n worker : {} \n queue_size: {}'.format(self.n_worker, self.queue_size))
        print('# worker : {}'.format(self.n_worker))
        print('# queue_size : {}'.format(self.queue_size))

    def training(self, train_dataprovider, test_dataprovider):
        """
        :param train_dataprovider: keras.utils.Sequence
         학습 시 사용합니다. 출력값으로 batch_xs, batch_ys 가 되는 keras Sequence Generator 을 반환합니다.
        :param test_dataprovider: keras.utils.Sequence
         평가 시 사용합니다. 출력값으로 batch_xs, batch_ys 가 되는 keras Sequence Generator 을 반환합니다.
        :return:
        """
        best_acc = 0
        self.generate_folder()
        try:
            for i in range(self.n_epochs):
                # train
                print('Step : {}\n'.format(i))

                # lr_scheduler가 None 이면 기본 optimizer 에서 지정했던 learning rate 가 적용됩니다.
                hist = self.model.fit(train_dataprovider,
                                      epochs=1,
                                      max_queue_size=self.queue_size, workers=self.n_worker,
                                      steps_per_epoch=self.step_per_epoch,
                                      callbacks=[self.lr_scheduler])

                train_loss = hist.history['loss'][0]
                train_acc = hist.history['acc'][0]
                lr = hist.history['lr'][0]

                # evaluation
                if test_dataprovider is None:
                    val_acc = train_acc
                    val_loss = train_loss
                else:
                    val_loss, val_acc = self.model.evaluate(test_dataprovider,
                                                            max_queue_size=self.queue_size,
                                                            workers=self.n_worker)
                # best accuracy
                if val_acc > best_acc or best_acc == 0:
                    best_acc = val_acc

                    # model save
                    self.best_model_path = os.path.join(self.model_dir, 'classifier_{}'.format(i))
                    self.model.save(self.best_model_path)
                print('Best Acc : {}'.format(best_acc))

                # write log
                self.log_writer.write("{}\t{}\t{}\t{}\t{}\n".format(train_loss, train_acc, val_loss, val_acc, lr))
                self.log_writer.flush()

        except Exception as e:
            print(e)
            self.log_writer.close()
        self.log_writer.close()


def generate_tmp_folder(folder_name):
    """
    Description:
    입력된 인자의 경로에 폴더가 존재한다면 뒤에 숫자를 추가합니다.
    foldername -> foldername_0
    Args:
        folder_name:
    Returns:
    """
    tmp_count = 0
    new_folder_name = copy(folder_name)
    while (True):
        if os.path.isdir(new_folder_name):
            # 폴더가 존재하면 count 을 1올립니다.
            tmp_count += 1
            new_folder_name = '{}_{}'.format(folder_name, tmp_count)
        else:
            # 폴더가 존재하지 않으면 새롭게 만들어진 폴더 name 을 반환하니다.
            return new_folder_name

