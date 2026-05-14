import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_input = "yolo26n_int8.onnx"
model_output = "yolo26n_int4.onnx"

# Tiến hành lượng tử hóa
quantize_dynamic(
    model_input=model_input,
    model_output=model_output,
    weight_type=QuantType.QFLOAT8E4M3FN
)

print(f"✅ Đã convert thành công sang: {model_output}")