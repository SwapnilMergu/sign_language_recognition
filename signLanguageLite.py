import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

folders=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# Step 1: Read the images and extract features
def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Resize the image to a fixed size
    flattened = resized.flatten()  # Convert 2D array to 1D
    return flattened

# Step 2: Load and preprocess the dataset
def load_dataset():
    image_path = "D:/Swapnil/python/tensorflow/Images/"
    labels = []  # 0 represents A, 25 represents Z

    # Extract features from images
    features = []
    for idx in range(0,26):
        #print(image_path+folders[idx]+"/")
        for i in range(1,101):
            features.append(extract_features(image_path+folders[idx]+"/"+str(i)+".png"))
            labels.append(idx)
    X = np.array(features)
    y = np.array(labels)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



# Step 3: Train the classifier
def train_classifier():
    X_train, X_test, y_train, y_test = load_dataset()
    classifier = make_pipeline(SVC())
    classifier.fit(X_train, y_train)
    return classifier



# Step 4: Convert the classifier to a TensorFlow model
def convert_to_tf_model(classifier):
    input_shape = (64 * 64,)  # Input shape based on the flattened image size
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    svm = classifier.named_steps['svc']

    if svm.kernel == 'linear':
        # Linear kernel case
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape(input_shape, input_shape=input_shape),
            tf.keras.layers.Dense(1, activation='linear', use_bias=False)
        ])
        model.layers[1].set_weights([svm.coef_.T])
    else:
        '''# Non-linear kernel case
        support_vectors = svm.support_vectors_
        dual_coefs = svm.dual_coef_

        # Calculate the weights using the support vectors and dual coefficients
        weights = np.dot(dual_coefs, support_vectors)
        weights = np.squeeze(weights)  # Remove the redundant dimension
        weights = weights.reshape((-1, 1))
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape(input_shape, input_shape=input_shape),
            tf.keras.layers.Dense(1, activation='linear', use_bias=False)
        ])
        model.layers[1].set_weights([weights])'''
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(100,)))
        
        # Increase the weight of the layer by adjusting kernel_initializer and/or kernel_regularizer
        model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

# Step 5: Convert the TensorFlow model to a TensorFlow Lite model
def convert_to_tflite_model(tf_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()
    return tflite_model

# Load and train the classifier
classifier = train_classifier()
print("training classifier")

'''
# Perform prediction on a new image
def predict_image(classifier, image_path):
    # Extract features from the new image
    features = extract_features(image_path)

    # Perform prediction
    prediction = classifier.predict([features])

    return prediction
new_image_path = 'D:/Swapnil/python/tensorflow/Images/F/103.png'  # Replace with the path to your new image
prediction = predict_image(classifier, new_image_path)
print(prediction)   
print("Predicted value:",folders[int(prediction)])
'''

# Convert the classifier to a TensorFlow model
tf_model = convert_to_tf_model(classifier)
print("converting classifier to tf_model")

# Convert the TensorFlow model to a TensorFlow Lite model
tflite_model = convert_to_tflite_model(tf_model)
print("converting the tf_model to tflite_mode")

# Save the TFLite model to a file
with open('D:/Swapnil/python/tensorflow/signLanguageTfliteModel.tflite', 'wb') as f:
    f.write(tflite_model)
print("tflite model created successfuly")