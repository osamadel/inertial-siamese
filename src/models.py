def build_siamese(input_shape, model, dropout_rate):
    '''
    input_shape (tuple): the shape of each input tensor
    model (tensorflow.keras.models.Sequential):
            the ANN shared between the two inputs of the 
            siamese network.
    dropout_rate (float): the dropout rate at the output
            layer.
    
    returns the siamese network model
    '''
    from tensorflow.keras.layers import Input, Lambda, Dropout, Dense
    from tensorflow.keras.models import Model
    import tensorflow as tf
    
    
    # Define the tensors for the two input images
    left_input = Input(input_shape, name='left_input')
    right_input = Input(input_shape, name='right_input')

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:tf.abs(tensors[0] - tensors[1]), name='lambda')
    L1_distance = L1_layer([encoded_l, encoded_r])
    dropt = Dropout(rate = dropout_rate)(L1_distance)
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', name='output')(dropt)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs = [left_input, 
                                    right_input], 
                        outputs=prediction)

    # return the siamese network
    return siamese_net


def build_triplet_nn(input_shape, model):
    '''
    input_shape (tuple): the shape of each input tensor
    model (tensorflow.keras.models.Sequential):
            the ANN shared between the two inputs of the 
            siamese network.
    dropout_rate (float): the dropout rate at the output
            layer.
    
    returns the siamese network model
    '''
    from tensorflow.keras.layers import Input, Lambda, Dropout, Dense, Concatenate
    from tensorflow.keras.models import Model
    import tensorflow as tf
    
    
    # Define the tensors for the two input images
    anchor = Input(input_shape, name='anchor')
    positive = Input(input_shape, name='positive')
    negative = Input(input_shape, name='negative')

    # Generate the encodings (feature vectors) for the two images
    encoded_a = model(anchor)
    encoded_p = model(positive)
    encoded_n = model(negative)
    merged_vector = Concatenate(axis=1)([encoded_a, encoded_p, encoded_n])

    # Add a customized layer to compute the absolute difference between the encodings
#     L1_layer = Lambda(lambda tensors:tf.abs(tensors[0] - tensors[1]), name='lambda')
#     L1_distance_p = L1_layer([encoded_a, encoded_p])
#     L1_distance_n = L1_layer([encoded_a, encoded_n])

    
    
#     triplet_nn = Model(inputs = [anchor, positive, negative], 
#                         outputs=[L1_distance_p, L1_distance_n])
    triplet_nn = Model(inputs = [anchor, positive, negative], 
                        outputs=merged_vector)

    # return the siamese network
    return triplet_nn


def triplet_loss(y_true, y_pred, m = 0.4):
        import tensorflow.keras.backend as K
        size = y_pred.shape[1] // 3
        encoded_a = y_pred[:,:size]
        encoded_p = y_pred[:,size:size*2]
        encoded_n = y_pred[:,size*2:size*3]
        positive_distance = K.sum(K.square(encoded_a-encoded_p), axis=1)
        negative_distance = K.sum(K.square(encoded_a-encoded_n), axis=1)
        loss = K.maximum(positive_distance - negative_distance + m, 0.0)
        return loss


def build_siamese_vis(input_shape, model, dropout_rate):
    '''
    input_shape (tuple): the shape of each input tensor
    model (tensorflow.keras.models.Sequential):
            the ANN shared between the two inputs of the 
            siamese network.
    dropout_rate (float): the dropout rate at the output
            layer.
    
    returns the siamese network model
    '''
    from tensorflow.keras.layers import Input, Lambda, Dropout, Dense
    from tensorflow.keras.models import Model
    import tensorflow as tf

    # Define the tensors for the two input images
    left_input = Input(input_shape, name='left_input')
    right_input = Input(input_shape, name='right_input')

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    dropt = Dropout(rate = dropout_rate)(L1_distance)
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', name='output')(dropt)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs = [left_input, 
                                    right_input], 
                        outputs=[prediction, L1_distance])

    # return the siamese network
    return siamese_net


def build_1DCNN_2layers(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPooling1D
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.layers import Dropout, BatchNormalization
    from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv1D(64, 5, strides=3, 
                    padding='valid', activation='relu', 
                    input_shape=input_shape))
    # model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 3, strides=2, 
                    padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    # model.add(BatchNormalization())
    model.add(Conv1D(128, 2, strides=1, 
                    padding='valid', activation='tanh'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(GlobalMaxPooling1D())
    # model.add(Dropout(rate=0.2))
    # model.add(Dense(128, activation='tanh'))

    return model


def build_1DCNN_2layers_BN(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPooling1D
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.layers import Dropout, BatchNormalization
    from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv1D(32, 5, strides=3, 
                    padding='valid', activation='relu', 
                    input_shape=input_shape))
    # model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 3, strides=2, 
                    padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    # model.add(BatchNormalization())
    model.add(Conv1D(128, 2, strides=1, 
                    padding='valid', activation='tanh'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(GlobalMaxPooling1D())
    # model.add(Dropout(rate=0.2))
    # model.add(Dense(128, activation='tanh'))

    return model