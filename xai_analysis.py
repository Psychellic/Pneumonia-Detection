import os

import lime
import lime.lime_image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model

IMG_SIZE = 128
LAST_CONV_LAYER = "conv2d_2"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def generate_gradcam(model, img_path, output_path, last_conv_layer_name="conv2d_2"):
    img_array = tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(
            img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale"
        )
    )
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[2], img_array.shape[1]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    img_array = img_array.squeeze()
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.uint8(255 * img_array)
    img_array = tf.keras.utils.array_to_img(img_array)
    img_array = tf.keras.utils.img_to_array(img_array)

    superimposed_img = jet_heatmap * 0.4 + img_array
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    tf.keras.utils.save_img(output_path, superimposed_img)

    print(f"Grad-CAM image saved to {output_path}")


def explain_with_lime(model, image_gray, image_rgb, top_labels=2, num_samples=100):
    def predict_grayscale(images):
        images_gray = np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])
        images_gray = np.expand_dims(images_gray, axis=-1)
        return model.predict(images_gray, verbose=0)

    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb.astype("double"),
        classifier_fn=predict_grayscale,
        top_labels=top_labels,
        num_samples=num_samples,
        batch_size=10,
        num_features=5,
        hide_color=0,
        random_seed=42,
    )
    return explanation


def generate_lime(model, img_array, img_array_rgb, output_path):
    explanation = explain_with_lime(model, img_array, img_array_rgb, top_labels=2)
    predicted_label = np.argmax(
        model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    )

    temp, mask = explanation.get_image_and_mask(
        predicted_label, positive_only=False, num_features=5, hide_rest=False
    )
    lime_image = mark_boundaries(img_array_rgb, mask)

    plt.imsave(output_path, lime_image)
    print(f"LIME image saved to {output_path}")


def generate_shap(model, img_array_expanded, output_path):
    num_background_samples = min(img_array_expanded.shape[0], 5)
    background = img_array_expanded[
        np.random.choice(
            img_array_expanded.shape[0], num_background_samples, replace=False
        )
    ]
    e = shap.GradientExplainer(model, background)
    shap_values, indexes = e.shap_values(img_array_expanded, ranked_outputs=2)

    shap_values_for_save = [shap_values[i] for i in range(len(shap_values))]

    fig, ax = plt.subplots()

    shap.image_plot(shap_values_for_save, img_array_expanded, show=False)

    plt.savefig(output_path, bbox_inches="tight")
    print(f"SHAP image saved to {output_path}")


def main():
    try:
        model = load_model(
            "best_model.keras",
            custom_objects={"focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.5)},
        )
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    img_path = "chest_xray/test/NORMAL/NORMAL2-IM-0051-0001.jpeg"
    class_names = ["Normal", "Pneumonia"]

    img = tf.keras.utils.load_img(
        img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)

    img_array_rgb = np.repeat(img_array, 3, axis=-1)

    base_filename = os.path.splitext(os.path.basename(img_path))[0]

    gradcam_output_path = f"{base_filename}_gradcam.png"
    generate_gradcam(model, img_path, gradcam_output_path, LAST_CONV_LAYER)

    lime_output_path = f"{base_filename}_lime.png"
    generate_lime(model, img_array, img_array_rgb, lime_output_path)

    shap_output_path = f"{base_filename}_shap.png"
    generate_shap(model, img_array_expanded, shap_output_path)

    original_img_path = f"{base_filename}_original.png"
    tf.keras.utils.save_img(original_img_path, img_array)
    print(f"Original image saved to {original_img_path}")


if __name__ == "__main__":
    main()
