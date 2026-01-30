

class BaseModel:
    def __init__(self, model, input_shape=(224, 224, 3), classes=1000, weights='imagenet', classifier_activation='softmax'):
        self.model = model(
                include_top=False,
                weights=weights,
                input_tensor=None,
                input_shape=input_shape,
                pooling=None,
                classes=classes,
                classifier_activation=classifier_activation
        )

    def train(self, data, labels, epochs=10, batch_size=32, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.fit(data, labels, epochs=epochs, batch_size=batch_size)

    def summary(self):
        self.model.summary()


    def create_layer(self):
        print("Creating base model layer...")