
# Define the CNN-LSTM model for multi-class classification
def create_cnnlstm_model(input_size, num_classes, hparams):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten, LSTM, TimeDistributed, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.metrics import Precision, Recall, F1Score

    modelInput = Input(shape=input_size)
    model = TimeDistributed(Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', padding="same"))(modelInput)
    model = TimeDistributed(Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', padding="same"))(model)
    model = TimeDistributed(MaxPooling1D(hparams["HP_POOL"]))(model)
    model = TimeDistributed(Flatten())(model)
    model = LSTM(hparams["HP_LSTM_UNITS"], return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = LSTM(hparams["HP_LSTM_UNITS"], return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dense(hparams["HP_H_UNITS"], activation='relu', kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dropout(0.5)(model)
    model = Dense(num_classes, activation='softmax')(model)  # Output layer with softmax for multi-class

    model = Model(modelInput, model)

    opt = tf.keras.optimizers.Adam(learning_rate=hparams["HP_LR"])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(average='macro')])
    
    print(model.summary())
    return model


# Define the CNN-LSTM model for multi-class classification
def create_lstm_model(input_size, num_classes, hparams):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.metrics import Precision, Recall, F1Score

    modelInput = Input(shape=input_size)
    model = LSTM(hparams["HP_LSTM_UNITS"], return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(modelInput)
    model = LSTM(hparams["HP_LSTM_UNITS"], return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dense(hparams["HP_H_UNITS"], activation='relu', kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dropout(0.5)(model)
    model = Dense(num_classes, activation='softmax')(model)  # Output layer with softmax for multi-class

    model = Model(modelInput, model)

    opt = tf.keras.optimizers.Adam(learning_rate=hparams["HP_LR"])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(average='macro')])
    
    print(model.summary())
    return model

# Define the CNN model for multi-class classification
def create_cnn_model(input_size, num_classes, hparams):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.metrics import Precision, Recall, F1Score

    modelInput = Input(shape=input_size)
    model = Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', padding="same")(modelInput)
    model = Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', padding="same")(model)
    model = MaxPooling1D(hparams["HP_POOL"])(model)
    model = Flatten()(model)
    model = Dense(hparams["HP_H_UNITS"], activation='relu', kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dropout(0.5)(model)
    model = Dense(num_classes, activation='softmax')(model)  # Output layer with softmax for multi-class

    model = Model(modelInput, model)

    opt = tf.keras.optimizers.Adam(learning_rate=hparams["HP_LR"])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(average='macro')])
    
    print(model.summary())
    return model

def create_svm_model():
    from sklearn import svm
    model = svm.SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    return model

def create_random_forest_model():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def create_logistic_regression_model():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42, max_iter=1000)
    return model

def create_knn_model():
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    return model

def create_naive_bayes_model():
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    return model

def create_decision_tree_model():
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)
    return model

