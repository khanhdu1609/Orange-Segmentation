from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, Callback
from tensorflow.keras.optimizers import Adam
class ELUnet:
  def __init__(self, num_filters=8, height = 256, width = 256):
    self.num_filters = num_filters
    self.inputs = Input(shape=(height, width, 3))
  def conv_block(self, inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)

    return x
  def encoder_block(self, input, num_filters):
    x = self.conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p 
  def build_elunet(self):
    #Encoder
    s1, p1 = self.encoder_block(self.inputs, self.num_filters)
    s2, p2 = self.encoder_block(p1, 2*self.num_filters)
    s3, p3 = self.encoder_block(p2, 4*self.num_filters)
    s4, p4 = self.encoder_block(p3, 8*self.num_filters)

    #Bridge
    b1 = self.conv_block(p4, 16*self.num_filters)

    #Decoder
    #d4
    d4 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(b1)
    s4_conv = Conv2D(64, 3, padding='same')(s4)
    s4_conv = BatchNormalization()(s4_conv)
    d4 = Concatenate()([d4, s4_conv])
    d4 = self.conv_block(d4, 8*self.num_filters)

    # d3
    d3 = Conv2DTranspose(32, (3, 3), strides=(2,2), padding='same')(d4)
    s4_upsamp2 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(s4)
    s3_conv = Conv2D(32, 3, padding='same')(s3)
    s3_conv = BatchNormalization()(s3_conv)
    d3 = Concatenate()([d3,s4_upsamp2, s3_conv])
    d3 = self.conv_block(d3, 4*self.num_filters)

    # d2
    d2 = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(d3)
    s4_upsamp4 = Conv2DTranspose(16, (3, 3), strides=4, padding='same')(s4)
    s3_upsamp2 = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(s3)
    s2_conv = Conv2D(16, 3, padding='same')(s2)
    s2_conv = BatchNormalization()(s2_conv)
    d2 = Concatenate()([d2, s4_upsamp4, s3_upsamp2, s2_conv])
    d2 = self.conv_block(d2, 2*self.num_filters)

    #d1
    d1 = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(d2)
    s4_upsamp8 = Conv2DTranspose(8, (3, 3), strides=8, padding='same')(s4)
    s3_upsamp4 = Conv2DTranspose(8, (3, 3), strides=4, padding='same')(s3)
    s2_upsamp2 = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(s2)
    s1_conv = Conv2D(8, 3, padding='same')(s1)
    s1_conv = BatchNormalization()(s1_conv)
    d1 = Concatenate()([d1, s4_upsamp8, s3_upsamp4, s2_upsamp2, s1_conv])
    d1 = self.conv_block(d1, self.num_filters)

    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d1)
    model = Model(self.inputs, outputs, name='ELU-net')
    return model
