import tensorflow as tf
from tensorflow.keras import layers, Model


class BaseModel:
    def __init__(self, model, input_shape=(224, 224, 3), weights='imagenet', classifier_activation='softmax', dense_units=256, output_activation='sigmoid', include_preprocessing=None):
        if(include_preprocessing != None):
            self.model = model(
                include_top=False,
                weights=weights,
                input_tensor=None,
                input_shape=input_shape,
                pooling=None,
                classifier_activation=classifier_activation,
                include_preprocessing=include_preprocessing
        )
        else:
            self.model = model(
                include_top=False,
                weights=weights,
                input_tensor=None,
                input_shape=input_shape,
                pooling=None,
                classifier_activation=classifier_activation,
        )
        outputlayer = self.model.output
        outputlayer = layers.GlobalAveragePooling2D()(outputlayer)
        outputlayer = layers.Dense(dense_units, activation=classifier_activation)(outputlayer)
        outputs = layers.Dense(1, activation=output_activation)(outputlayer)
        self.model = Model(inputs=self.model.input, outputs=outputs)







    def train(self, data, labels, epochs=10, batch_size=32, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.fit(data, labels, epochs=epochs, batch_size=batch_size)

    def summary(self):
        self.model.summary()


    def create_layer(self):
        print("Creating base model layer...")