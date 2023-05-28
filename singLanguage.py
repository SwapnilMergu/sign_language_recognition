import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

x_train=0
# Step 1: Read the images and extract features
def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Resize the image to a fixed size
    flattened = resized.flatten()  # Convert 2D array to 1D
    return flattened

# Step 2: Load and preprocess the dataset

def load_dataset():
    # List of image paths and corresponding labels
    image_path = "D:/Swapnil/python/tensorflow/Images/" #['cat1.jpg', 'cat2.jpg', 'dog1.jpg', 'dog2.jpg']
    labels = []  # 0 represents cat, 1 represents dog

    # Extract features from images
    features = []

    for file in ["A","Z"]:
        for i in range(1,101):
            features.append(extract_features(image_path+file+"/"+str(i)+".png"))
            if(file=="A"):
                labels.append(0)
            else:
                labels.append(1)
    x = np.array(features)
    y = np.array(labels)

#   for image_path in image_paths:
#        features.append(extract_features(image_path))


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train=X_train
    return X_train, X_test, y_train, y_test,x,y
    #return x,y

def predict_image(classifier, image_path):
    # Extract features from the new image
    features = extract_features(image_path)

    # Perform prediction
    prediction = classifier.predict([features])

    return prediction


def train_classifier():
    # Load the dataset
    X_train, X_test, y_train, y_test, x, y = load_dataset()

    # Create an SVM classifier
    classifier = SVC()

    # Train the classifier
    classifier.fit(x, y)

    # converter=tf.lite.TFLiteConverter.from_keras_model(classifier)
    # tflite_model=converter.convert()

    # with open("model.tflite","wb") as f:
    #     f.write(tflite_model)


    return classifier


# Load and train the classifier
#classifier = train_classifier()


# Step 3: Train and evaluate the classifier
def train_and_evaluate():
    # Load the dataset
    X_train, X_test, y_train, y_test,x,y = load_dataset()
    print("X_train:",X_train,"\nX_test",X_test)
    print("X_train:",X_train,"\nX_test",X_test)
    print(X_test.shape)
    # Create an SVM classifier
    classifier = SVC()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict the labels for test set
    # image = cv2.imread("D:/Swapnil/python/tensorflow/117.png")
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(gray, (64, 64))  # Resize the image to a fixed size
    # flattened = resized.flatten()
    # res=[np.array(flattened)]
    #print(flattened.reshape(1,2))
    y_pred = classifier.predict(X_test)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Run the training and evaluation
train_and_evaluate()


'''new_image_path = 'D:/Swapnil/python/tensorflow/demo.png'  # Replace with the path to your new image
prediction = predict_image(classifier, new_image_path)

# Print the predicted label
if prediction == 0:
    print("Predicted: A")
elif prediction == 1:
    print("Predicted: Z")
else:
    print("Unknown prediction")'''




def convert_to_tf_model(classifier):
    # Create a scikit-learn pipeline with a StandardScaler and the SVC classifier
    model = make_pipeline(StandardScaler(), classifier)
    
    # Convert the scikit-learn pipeline to a TensorFlow model
    input_signature = [
        tf.TensorSpec(shape=(None, x_train.shape[1]), dtype=tf.float32)
    ]
    tf_model = tf.keras.wrappers.scikit_learn.KerasClassifier(model).model
    tf_model._saved_model_inputs_spec = input_signature
    
    return tf_model

# Step 5: Convert the TensorFlow model to a TensorFlow Lite model
def convert_to_tflite_model(tf_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()
    return tflite_model

# Load and train the classifier
classifier = train_classifier()

# Convert the classifier to a TensorFlow model
tf_model = convert_to_tf_model(classifier)

# Convert the TensorFlow model to a TensorFlow Lite model
tflite_model = convert_to_tflite_model(tf_model)

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)