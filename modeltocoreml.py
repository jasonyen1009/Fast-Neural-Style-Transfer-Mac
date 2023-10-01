from models import TransformerNet
import torch
import argparse
import os
from utils import *
import coremltools as ct
from coremltools.proto import FeatureTypes_pb2 as ft
from coremltools.models.utils import save_spec
from coremltools.models.neural_network import quantization_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    args = parser.parse_args()
    print(args)

    transform = style_transform()

    device = "mps" if getattr(torch,'has_mps',False) \
    else "gpu" if torch.cuda.is_available() else "cpu"

    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(args.checkpoint_model, map_location=torch.device('cpu')))
    transformer.eval()
    print(device)

    # pre-trained/starry_night_10000.pth -> starry_night
    name = args.checkpoint_model.split("/")
    print(name) # ['pre-trained', 'starry_night_10000.pth']
    name = name[-1] # ['starry_night_10000.pth']

    if name == "starry_night_10000.pth":
        name = "starry_night"
    elif name == "mosaic_10000.pth":
        name = "mosaic"
    elif name == "cuphead_10000.pth":
        name = "cuphead"

    # 'starry_night_10000.pth' -> 'starry_night'
    # name = name.split("_")[0] + "_" + name.split("_")[1]


    example_input = torch.randn(1,3,960,640).to(device)
    traced_model = torch.jit.trace(transformer, example_input)

    mlmodel = ct.convert(traced_model, inputs=[ct.ImageType(name="input", shape=example_input.shape,bias=[-0.485/0.229,-0.456/0.224,-0.406/0.225],scale=1.0/255.0/0.226)])

    if args.fp16:
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)
    elif args.int8:
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8)


    spec = mlmodel.get_spec()
    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
    builder.spec.description

    # var_294
    output_name = spec.description.output[0].name

    builder.add_split(name="split", input_name=output_name, output_names=["split_1","split_2","split_3"])
    builder.add_activation(name="activation_1",non_linearity="LINEAR",input_name="split_1",output_name="activation_out_1",params=[255*0.226,0.485*255])
    builder.add_activation(name="activation_2",non_linearity="LINEAR",input_name="split_2",output_name="activation_out_2",params=[255*0.226,0.456*255])
    builder.add_activation(name="activation_3",non_linearity="LINEAR",input_name="split_3",output_name="activation_out_3",params=[255*0.226,0.406*255])
    builder.add_stack(name="stack", input_names=["activation_out_1","activation_out_2","activation_out_3"], output_name="stack_out", axis=0)
    builder.add_squeeze(name="squeeze", input_name="stack_out", output_name="squeeze_out", axes = None, squeeze_all = True)


    builder.spec.description.output.pop()
    builder.spec.description.output.add()
    output = builder.spec.description.output[0]
    output.name = "squeeze_out"

    output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
    output.type.imageType.width = 640 
    output.type.imageType.height = 960

    if args.fp16:
        save_spec(builder.spec, '{}_fp16.mlmodel'.format(name))
        print("FP16 model saved.")
    elif args.int8:
        save_spec(builder.spec, '{}_int8.mlmodel'.format(name))
        print("INT8 model saved.")
    else:
        save_spec(builder.spec, '{}.mlmodel'.format(name))
        print("FP32 model saved.")
