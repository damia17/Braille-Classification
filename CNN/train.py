from keras.preprocessing.image import ImageDataGenerator
from keras import backend as kr
from keras import layers as layer
from keras.models import Model,load_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping


#? Data Generator
data = ImageDataGenerator(rotation_range=20,
                             shear_range=10,
                             validation_split=0.2)

train_gen = data.flow_from_directory('./dataImages/',
                                              target_size=(28,28),
                                              subset='training')

val_gen = data.flow_from_directory('./dataImages/',
                                            target_size=(28,28),
                                            subset='validation')

kr.clear_session()

chkpt = ModelCheckpoint('BrailleClassification.h5',save_best_only=True)
rlr = ReduceLROnPlateau(patience=8,verbose=0)
early_stop = EarlyStopping(patience=15,verbose=1)

#? Model
def create_model(filters1=64, filters2=128, filters3=256, kernel_size=(3,3), learning_rate=1e-3):
    entry = layer.Input(shape=(28, 28, 3))
    x = layer.SeparableConv2D(filters1, kernel_size, activation='relu')(entry)
    x = layer.MaxPooling2D((2, 2))(x)
    x = layer.SeparableConv2D(filters2, kernel_size, activation='relu')(x)
    x = layer.MaxPooling2D((2, 2))(x)
    x = layer.SeparableConv2D(filters3, (2, 2), activation='relu')(x)
    x = layer.GlobalMaxPooling2D()(x)
    x = layer.Dense(256)(x)
    x = layer.LeakyReLU()(x)
    x = layer.Dense(64, kernel_regularizer=l2(2e-4))(x)
    x = layer.LeakyReLU()(x)
    output = layer.Dense(26, activation='softmax')(x)

    model = Model(inputs=entry, outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


