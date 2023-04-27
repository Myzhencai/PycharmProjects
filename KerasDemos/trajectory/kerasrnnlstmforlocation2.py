import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# generate sine wavepip
def make_sine_with_noise(_start, _stop, _step, _phase_shift, gain):
    x = numpy.arange(_start, _stop, step = _step)
    noise = numpy.random.uniform(-0.1, 0.1, size = len(x))
    y = gain*0.5*numpy.sin(x+_phase_shift)
    y = numpy.add(noise, y)
    return x, y
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_ahead=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_ahead - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        b = dataset[(i + look_back):(i + look_back + look_ahead), :]
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# generate sine wave
x1, y1 = make_sine_with_noise(0, 200, 1/24, 0, 1)
x2, y2 = make_sine_with_noise(0, 200, 1/24, math.pi/4, 3)
x3, y3 = make_sine_with_noise(0, 200, 1/24, math.pi/2, 20)
# plt.plot(x1, y1)
# plt.plot(x2, y2)
# plt.plot(x3, y3)
# plt.show()
#transform to pandas dataframe
dataframe = pandas.DataFrame({'y1': y1, 'y2': y2, 'x3': y3})
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 10
look_ahead = 5
trainX, trainY = create_dataset(train, look_back, look_ahead)
testX, testY = create_dataset(test, look_back, look_ahead)
print(trainX.shape)
print(trainY.shape)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(look_ahead, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(look_ahead, input_shape=(look_ahead, trainX.shape[2])))
model.add(Dense(trainY.shape[1]*trainY.shape[2]))
model.add(Reshape((trainY.shape[1], trainY.shape[2])))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=1)
# make prediction
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#save model
model.save('my_sin_prediction_model.h5')

trainPredictPlottable = trainPredict[::look_ahead]
trainPredictPlottable = [item for sublist in trainPredictPlottable for item in sublist]
trainPredictPlottable = scaler.inverse_transform(numpy.array(trainPredictPlottable))
# create single testPredict array concatenating every 'look_ahed' prediction array
testPredictPlottable = testPredict[::look_ahead]
testPredictPlottable = [item for sublist in testPredictPlottable for item in sublist]
testPredictPlottable = scaler.inverse_transform(numpy.array(testPredictPlottable))
# testPredictPlottable = testPredictPlottable[:-look_ahead]
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredictPlottable)+look_back, :] = trainPredictPlottable
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(dataset)-len(testPredictPlottable):len(dataset), :] = testPredictPlottable
# plot baseline and predictions
dataset = scaler.inverse_transform(dataset)
plt.plot(dataset, color='k')
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
