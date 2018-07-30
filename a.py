



from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(1568, input_shape=(784,), activation='relu', kernel_initializer='he_normal'))
#model.add(Dropout(0.3))
#model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
#model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.1)

pred_y = model.predict(x_test)
pred_y = np.argmax(pred_y, 1)
    
submission = pd.Series(pred_y, name='label')
submission.to_csv('/root/userspace/submission.csv', header=True, index_label='id')

