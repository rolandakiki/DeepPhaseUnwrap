import numpy as np
from keras.optimizers import Adam

# internal 
from src.data.utils import load_phase_data
from src.models.architectures import JointConvSQDLSTMNet

# ////// body /////

## user inputs
use_JointConvSQDLSTMNet = True
model_id = "Model_JointConvSQDLSTMNet_Noisy_Phase_Data_1000_8pi_8pi"
dataset_id = "Noisy_Phase_Data_400_8pi_8pi" # testing dataset
image_size = (256, 256, 1) 
batch_size = 4

## load test data and visualize a pair
X_test, y_test = load_phase_data(dataset_id)
print("Number of Testing Samples : {:n}".format(X_test.shape[0]))
idx = np.random.randint(0, X_test.shape[0])

## load trained model
if use_JointConvSQDLSTMNet:
    model = JointConvSQDLSTMNet(image_size).getModel()
    model.summary()

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    metrics=[]
)

model_path = './DeepPhaseUnwrap/models/{}.h5'.format(model_id)
model.load_weights(model_path)

## predict true phase
y_pred = model.predict(X_test, batch_size=batch_size)

## get the scaled predicted true phase values
y_pred_scaled = np.empty((0, 256, 256))
for i in range(X_test.shape[0]):
  Xi = X_test[i]
  yi = y_test[i]
  ypi = y_pred[i]
  
  # match scales of predicted true phase
  min1, max1 = np.min(yi), np.max(yi)
  min2, max2 = np.min(ypi), np.max(ypi)
  temp = (ypi - min2) / (max2 - min2)
  ypi_scaled = temp * (max1 - min1) + min1
  y_pred_scaled = np.vstack((y_pred_scaled, ypi_scaled.reshape(1, 256, 256)))

## compute Normalize Root Mean Squared Error
error = y_test - y_pred_scaled
r = np.max(y_test, axis=(1, 2), keepdims=True) - np.min(y_test, axis=(1, 2), keepdims=True)
NRMSE = np.mean(np.sqrt(np.mean(error**2, axis=(1, 2)))/r)*100
performance = "NRMSE = {:.2f} %".format(NRMSE)
print(performance)
