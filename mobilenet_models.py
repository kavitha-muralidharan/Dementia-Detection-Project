import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Paths to datasets
train_dir = '/home/saivarun/alzhemeir_detection_work/Combined Dataset/train'
test_dir = '/home/saivarun/alzhemeir_detection_work/Combined Dataset/test'

# Parameters
image_size = (224, 224)  # Resize to 224x224
batch_size = 32
num_classes = 4
epochs = 15

# Data Rescaling Only
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Function to save plots
def save_plot(filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")

# Function to plot accuracy and loss
def plot_accuracy_loss(history, model_name):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(epochs_range, history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')

    save_plot(f"{model_name}_Accuracy_Loss.png")
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')

    save_plot(f"{model_name}_Confusion_Matrix.png")
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true_onehot, y_pred, model_name, class_names):
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_name} (AUC = {roc_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)', linewidth=2)
    plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.grid()

    save_plot(f"{model_name}_ROC_Curve.png")
    plt.show()

# Function to train model
def train_model(base_model, model_name, train_gen, test_gen):
    # Load the pre-trained model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_gen, validation_data=test_gen, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Evaluate the model
    test_generator.reset()
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred_classes, target_names=test_gen.class_indices.keys()))

    # Plot accuracy and loss
    plot_accuracy_loss(history, model_name)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, model_name, test_gen.class_indices.keys())

    # ROC curve
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
    plot_roc_curve(y_true_onehot, y_pred, model_name, test_gen.class_indices.keys())
    
    return history.history['val_accuracy'][-1]  # Return final validation accuracy

# Load models and train
models = {
    "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "MobileNetV3": MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
}

accuracies = {}

for model_name, base_model in models.items():
    print(f"\nTraining {model_name}...\n")
    val_accuracy = train_model(base_model, model_name, train_generator, test_generator)
    accuracies[model_name] = val_accuracy

# Comparative analysis
def plot_comparative_analysis(accuracies):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='Set2')
    plt.title("Validation Accuracy Comparison")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Models")
    save_plot("Model_Comparison.png")
    plt.show()

plot_comparative_analysis(accuracies)
