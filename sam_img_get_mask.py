import argparse
import os.path
import time

import numpy as np
import onnxruntime
from PIL import Image


def predict_onnx(model_path, decoder_path, input_image, input_points, input_labels):
    model_name = os.path.basename(model_path)
    if model_name == "efficient_sam_vitt.onnx":
        inference_session = onnxruntime.InferenceSession(
            model_path
        )

        (
            predicted_logits,
            predicted_iou,
            predicted_lowres_logits,
        ) = inference_session.run(
            output_names=None,
            input_feed={
                "batched_images": input_image,
                "batched_point_coords": input_points,
                "batched_point_labels": input_labels,
            },
        )
    else:
        inference_session = onnxruntime.InferenceSession(
            model_path
        )
        image_embeddings, = inference_session.run(
            output_names=None,
            input_feed={
                "batched_images": input_image,
            },
        )
        inference_session = onnxruntime.InferenceSession(
            decoder_path
        )

        (
            predicted_logits,
            predicted_iou,
            predicted_lowres_logits,
        ) = inference_session.run(
            output_names=None,
            input_feed={
                "image_embeddings": image_embeddings,
                "batched_point_coords": input_points,
                "batched_point_labels": input_labels,
                "orig_im_size": np.array(input_image.shape[2:], dtype=np.int64),
            },
        )
    mask = predicted_logits[0, 0, 0, :, :] >= 0
    return mask


def main():
    parser = argparse.ArgumentParser(description="ONNX model prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--decoder_path", type=str, default="efficient_sam_vitt_decoder.onnx", help="Model type")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--points", type=str, required=True, help="Input points as a list, e.g., '580,350;650,350'")
    parser.add_argument("--labels", type=str, required=True, help="Input labels as a list, e.g., '1,1'")
    parser.add_argument("--output_path", type=str, default=None, required=False, help="Path to save the output mask")

    args = parser.parse_args()
    model_path = args.model_path
    decoder_path = args.decoder_path
    image = np.array(Image.open(args.image_path))

    input_image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    points_string = args.points.replace('"', '').replace("'", "")
    points = [list(map(int, p.split(','))) for p in points_string.split(';')]
    input_points = np.array([[points]], dtype=np.float32)

    labels_string = args.labels.replace('"', '').replace("'", "")
    labels = list(map(int, labels_string.split(',')))
    input_labels = np.array([[labels]], dtype=np.float32)
    save_path = args.output_path
    mask = predict_onnx(model_path, decoder_path, input_image, input_points, input_labels)
    if save_path is not None:
        mask_image = np.uint8(mask) * 255
        image = Image.fromarray(mask_image)
        image.save(save_path)
    return mask


if __name__ == "__main__":
    t1 = time.time()
    mask = main()
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")
