"""Export the pair classifier to ONNX for CPU serving."""

import argparse
from pathlib import Path

import numpy as np
import torch

from dedupe.config import settings
from dedupe.training.model import PairClassifier


class _WithSigmoid(torch.nn.Module):
    """Export calibrated probabilities, not logits — threshold semantics stay identical."""

    def __init__(self, model: PairClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x) / self.model.temperature)


def main():
    ap = argparse.ArgumentParser(description="Export pair classifier to ONNX")
    ap.add_argument("--checkpoint", type=Path, default=Path(settings.models_dir) / "pair_classifier.pt")
    ap.add_argument("--out", type=Path, default=Path(settings.models_dir) / "pair_classifier.onnx")
    ap.add_argument("--int8", action="store_true", help="also write a dynamically-quantized copy")
    args = ap.parse_args()

    model = PairClassifier.load(args.checkpoint).eval()
    dim = model.config.dim
    x = torch.rand(2, dim)

    batch = torch.export.Dim("batch")
    torch.onnx.export(
        _WithSigmoid(model),
        (x,),
        str(args.out),
        input_names=["features"],
        output_names=["prob"],
        dynamo=True,
        dynamic_shapes={"features": {0: batch}},
        opset_version=18,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    x_np = np.random.rand(64, dim).astype(np.float32)
    ort_out = sess.run(None, {"features": x_np})[0]
    torch_out = model.predict(torch.from_numpy(x_np)).numpy()
    err = float(np.abs(ort_out.squeeze() - torch_out).max())
    if err > 1e-4:
        raise SystemExit(f"parity check failed: max |torch - ort| = {err:.2e}")
    print(f"{args.out}: parity ok (max err {err:.2e})")

    if args.int8:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # dynamo leaves value_info shape annotations that trip quantizer shape inference
        m = onnx.load(str(args.out))
        del m.graph.value_info[:]
        pre = args.out.with_suffix(".pre.onnx")
        onnx.save(m, str(pre))
        q8 = args.out.with_suffix(".int8.onnx")
        quantize_dynamic(str(pre), str(q8), weight_type=QuantType.QInt8)
        pre.unlink(missing_ok=True)
        s8 = ort.InferenceSession(str(q8), providers=["CPUExecutionProvider"])
        drift = float(np.abs(s8.run(None, {"features": x_np})[0].squeeze() - torch_out).max())
        print(f"{q8}: max prob drift vs fp32 {drift:.4f}")


if __name__ == "__main__":
    main()
