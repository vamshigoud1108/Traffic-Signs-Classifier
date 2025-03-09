## Traffic Signs Classifier
![](https://automaticaddison.com/wp-content/uploads/2021/02/self-driving-car-road-sign-detection.jpg)
## Project Overview
This project builds a Convolutional Neural Network (CNN) to classify traffic signs using labeled image data. The model is trained using TensorFlow and Keras, leveraging image preprocessing and augmentation techniques.

## Dataset
- The dataset contains images of various traffic signs.
- Missing class names were updated for better labeling.
- Images were rescaled and split into training and validation sets.

## Model Archictecture
- **3 Convolutional Layers** with ReLU activation and MaxPooling.
- **Flatten Layer** to convert feature maps into a vector.
- **Dense Layer** with dropout to prevent overfitting.
- **Softmax Output Layer** for classification.

## Training and Evaluation
- The model was trained using sparse_categorical_crossentropy loss and the Adam optimizer.
- Training and validation accuracy trends showed effective learning.
- Further improvements can be made with data augmentation and hyperparameter tuning.

## Future Improvements
- Experiment with more CNN layers and filters.
- Use data augmentation to improve generalization.
- Fine-tune hyperparameters for better performance.
