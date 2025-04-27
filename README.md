# segment_anything_via_onnx

## Description
When the segmentation service needs to be invoked in the program, the package size is too large. Therefore, the model is converted to ONNX format and inference is performed using ONNX Runtime.

### 1. Directory Structure

- `models` stores the original model files.  
- `onnx` stores the ONNX format models.  
- Two scripts:
  - `sam_img_get_mask.py`: Code for segmenting an image by passing points and labels.
  - `sam_everything.py`: Code for segmenting the entire image.

### 2. Environment

#### 2.1 Dependencies

```commandline
scipy                     1.15.1
onnx                      1.17.0
onnxruntime               1.21.0
opencv-python             4.11.0.86
```

### 3. Method to Package as EXE

```commandline
pyinstaller -F .\sam_everything.py
```

### 4. How to Invoke the EXE
```commandline
.\sam_img_get_mask.exe --model_path "D:\esam\EfficientSAM\weights\efficient_sam_vitt_encoder.onnx" --image_path "D:\esam\IMG_20250206_162736.jpg" --decoder_path "D:\esam\EfficientSAM\weights\efficient_sam_vitt_decoder.onnx" --points "580,350;650,350" --labels "1,1" --output_path "D:\esam\EfficientSAM\dist\mask.png"
```

#### Parameter Description:
- EXE Path: `.\sam_img_get_mask.exe`
- Encoder Path: `--model_path "D:\esam\EfficientSAM\weights\efficient_sam_vitt_encoder.onnx"`
- Decoder Path: `--decoder_path "D:\esam\EfficientSAM\weights\efficient_sam_vitt_decoder.onnx"`
- Image Info: `--image_path "D:\esam\IMG_20250206_162736.jpg"`
- Points Info: `--points "580,350;650,350"`
- Class Info: `--labels "1,1"`
- Save Info: `--output_path "D:\esam\EfficientSAM\dist\mask.png"` (Selected mask region is 255, other regions are 0.)

```commandline
.\sam_everything.exe --model_path D:\esam\FastSam_Awsome_TensorRT\FastSAM-x.onnx --image_path D:\esam\FastSAM\images\cat.jpg --output_mask_path D:\esam\EfficientSAM\dist\all_mask.png --output_json_path D:\esam\EfficientSAM\dist\all_mask.json
```

#### Parameter Description:
- EXE Path: `.\sam_everything.exe`
- Model Path: `--model_path D:\esam\FastSam_Awsome_TensorRT\FastSAM-x.onnx`
- Image Info: `--image_path D:\esam\FastSAM\images\cat.jpg`
- Output Mask Image: `--output_mask_path D:\esam\EfficientSAM\dist\all_mask.png`
- Output JSON Info: `--output_json_path D:\esam\EfficientSAM\dist\all_mask.json`

The JSON file stores the mapping between class and color values, allowing you to obtain the corresponding mask based on the colors.

### 5. References
> https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT/tree/main  
> 
> https://github.com/yformer/EfficientSAM


## 说明
在程序中需要调用分割服务但是，打包过大，因此将模型转换为onnx格式，使用onnxruntime进行推理。
### 1. 目录结构

models中存储原始model文件   
onnx存储onnx格式的模型   
两个代码，其中`sam_img_get_mask.py`是对图片通过传入点和label进行分割的代码，`sam_everything.py`是对图像进行分割的代码。

### 2. 环境

#### 2.1 依赖库

```commandline
scipy                     1.15.1
onnx                      1.17.0
onnxruntime               1.21.0
opencv-python             4.11.0.86
```


### 3. 打包exe方法

` pyinstaller -F .\sam_everything.py`



### 4. 调用exe方法
```commandline

.\sam_img_get_mask.exe --model_path "D:\esam\EfficientSAM\weights\efficient_sam_vitt_encoder.onnx" --image_path "D:\esam\IMG_20250206_162736.jpg" --decoder_path "D:\esam\EfficientSAM\weights\efficient_sam_vitt_decoder.onnx" --points "580,350;650,350" --labels "1,1"   --output_path "D:\esam\EfficientSAM\dist\mask.png"


参数说明
exe路径 .\sam_img_get_mask.exe 
编码器路径  --model_path "D:\esam\EfficientSAM\weights\efficient_sam_vitt_encoder.onnx" 
解码器路径 --decoder_path "D:\esam\EfficientSAM\weights\efficient_sam_vitt_decoder.onnx"
图像信息  --image_path "D:\esam\IMG_20250206_162736.jpg"
点的信息 --points "580,350;650,350"
类别信息   --labels "1,1" 
保存信息 --output_path "D:\esam\EfficientSAM\dist\mask.png" （选择的mask区域是255，其他区域是0）
```


```commandline

调用方式
.\sam_everything.exe --model_path D:\esam\FastSam_Awsome_TensorRT\FastSAM-x.onnx --image_path D:\esam\FastSAM\images\cat.jpg --output_mask_path D:\esam\EfficientSAM\dist\all_mask.png --output_json_path D:\esam\EfficientSAM\dist\all_mask.json

参数说明
exe路径 .\sam_everything.exe 
模型路径  --model_path D:\esam\FastSam_Awsome_TensorRT\FastSAM-x.onnx 
图像信息  --image_path D:\esam\FastSAM\images\cat.jpg
输出的mask图 --output_mask_path D:\esam\EfficientSAM\dist\all_mask.png
输出的json信息   --output_json_path D:\esam\EfficientSAM\dist\all_mask.json 

json中存储类别和颜色值之间的映射关系，可以根据色彩获取对应的mask

```


### 5. 参考资料
> https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT/tree/main
> 
> 
> https://github.com/yformer/EfficientSAM

