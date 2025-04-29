import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from IPython.display import display
import ipywidgets as widgets

# --------------------------
# Directory Setup
# --------------------------
save_dir = '/home/saivarun/alzhemeir_detection_work/plots'
os.makedirs(save_dir, exist_ok=True)

# --------------------------
# Model Load
# --------------------------
model_path = '/home/saivarun/alzhemeir_detection_work/saved_models/cnn_model.h5'
model = load_model(model_path)

# --------------------------
# Class Labels
# --------------------------
class_labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# --------------------------
# Image Upload Widget
# --------------------------
upload = widgets.FileUpload(accept='image/*', multiple=False)
display(upload)

def handle_upload(change):
    if upload.value:
        # Load uploaded image
        uploaded_filename = list(upload.value.keys())[0]
        content = upload.value[uploaded_filename]['content']
        img_path = os.path.join('/tmp', uploaded_filename)
        with open(img_path, 'wb') as f:
            f.write(content)

        run_explanations(img_path)

upload.observe(handle_upload, names='value')

# --------------------------
# Explanation Runner
# --------------------------
def run_explanations(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]

    # Plot prediction
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.savefig(f'{save_dir}/prediction.png', bbox_inches='tight')
    plt.show()

    # --------------------------
    # SHAP Explanation
    # --------------------------
    background = np.zeros((1, 224, 224, 3))  # Dummy background
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(img_array)
    shap.image_plot(shap_values, img_array, show=False)
    plt.savefig(f"{save_dir}/shap_explanation.png", bbox_inches='tight')
    plt.show()

    # --------------------------
    # LIME Explanation
    # --------------------------
    lime_explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        return model.predict(images)

    explanation = lime_explainer.explain_instance(
        img_array[0].astype('double'),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        label=predicted_class,
        positive_only=True,
        hide_rest=False,
        num_features=5,
        min_weight=0.0
    )

    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title('LIME Explanation')
    plt.axis('off')
    plt.savefig(f"{save_dir}/lime_explanation.png", bbox_inches='tight')
    plt.show()

    # --------------------------
    # Occlusion Map
    # --------------------------
    def occlusion_map(model, image, patch_size=20):
        image_copy = image.copy()
        height, width, _ = image.shape
        heatmap = np.zeros((height, width))

        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                occluded = image_copy.copy()
                occluded[i:i+patch_size, j:j+patch_size] = 0
                prediction = model.predict(np.expand_dims(occluded, axis=0))[0][predicted_class]
                heatmap[i:i+patch_size, j:j+patch_size] = prediction

        plt.imshow(heatmap, cmap='hot')
        plt.title('Occlusion Map')
        plt.axis('off')
        plt.savefig(f"{save_dir}/occlusion_map.png", bbox_inches='tight')
        plt.show()

    # Run Occlusion Map
    occlusion_map(model, img_array[0])

