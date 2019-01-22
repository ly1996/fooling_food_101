import os
import sys
import glob
import argparse
#import matplotlib.pyplot as plt

#from tensorflow import keras

from tensorflow.keras import __version__
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

FC_SIZE = 1024
FINE_TUNE_LAYER = 142
FINE_TUNE_LAYER_2 = 80
FINE_TUNE_LAYER_3 = 38
# RUN = RUN + 1 if 'RUN' in locals() else 1
LOG_DIR = os.path.expanduser('../data')
MODEL_FILE_PATH = os.path.expanduser('../data/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5')   # 模型Log文件以及.h5模型文件存放地址

# 冻上5a之前的层
def setup_to_finetune_1(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:FINE_TUNE_LAYER]:
     layer.trainable = False
  for layer in model.layers[FINE_TUNE_LAYER:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def setup_to_finetune_2(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:FINE_TUNE_LAYER_2]:
     layer.trainable = False
  for layer in model.layers[FINE_TUNE_LAYER_2:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def setup_to_finetune_3(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:FINE_TUNE_LAYER_3]:
     layer.trainable = False
  for layer in model.layers[FINE_TUNE_LAYER_3:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def setup_to_finetune_4(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


# 添加新层
def add_new_last_layer(base_model, nb_classes):
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model

# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

def main():
    global Width, Height, pic_dir_out, pic_dir_data
    output_model_file = os.path.expanduser("../data/model.h5")
    Width = 224
    Height = 224
    train_dir = os.path.expanduser('../images')  # 训练集数据
    val_dir = os.path.expanduser('../test-set')  # 验证集数据

    nb_classes = 101
    nb_epoch = 5
    batch_size = 100

    nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
    nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
    nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数
    nb_epoch = int(nb_epoch)  # epoch数量
    batch_size = int(batch_size)

    # 　图片生成器
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # 训练数据与测试数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(Width, Height),
        batch_size=batch_size, class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(Width, Height),
        batch_size=batch_size, class_mode='categorical')

    base_model = ResNet50(weights='imagenet', include_top=False)  # 预先要下载no_top模型
    model = add_new_last_layer(base_model, nb_classes)  # 从基本no_top模型上添加新层
    setup_to_transfer_learn(model, base_model)  # 冻结base_model所有层

    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    tensorboard = TensorBoard(log_dir=LOG_DIR, write_images=True)
    checkpoint = ModelCheckpoint(filepath=MODEL_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
    model.fit_generator(
        train_generator,
        steps_per_epoch=638,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=30,
        class_weight='auto',
        callbacks=[tensorboard, checkpoint]
        ,verbose = 1
    )

    # 设置网络结构
    setup_to_finetune_1(model)
    nb_epoch = 15

    # 模式二训练
    model.fit_generator(
        train_generator,
        steps_per_epoch=638,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=30,
        class_weight='auto',
        callbacks=[tensorboard, checkpoint],
        verbose = 1,
        initial_epoch=5
    )

    # 设置网络结构
    setup_to_finetune_2(model)
    nb_epoch = 30

    # 模式三训练
    model.fit_generator(
        train_generator,
        steps_per_epoch=638,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=30,
        class_weight='auto',
        callbacks=[tensorboard, checkpoint],
        verbose=1,
        initial_epoch=15
    )

    # 设置网络结构
    setup_to_finetune_3(model)
    nb_epoch = 50



    # 模式四训练
    model.fit_generator(
        train_generator,
        steps_per_epoch=638,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=30,
        class_weight='auto',
        callbacks=[tensorboard, checkpoint],
        verbose=1,
        initial_epoch=30
    )

    # 设置网络结构
    setup_to_finetune_4(model)
    nb_epoch = 300
    train_generator.batch_size = 70
    validation_generator.batch_size = 70

    # 模式五训练
    model.fit_generator(
        train_generator,
        steps_per_epoch=900,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=40,
        class_weight='auto',
        callbacks=[tensorboard, checkpoint],
        verbose=1,
        initial_epoch=50
    )

    # 模型保存
    model.save(output_model_file)

if __name__ == '__main__':
    main()
