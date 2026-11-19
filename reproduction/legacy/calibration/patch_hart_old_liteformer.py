"""Restore the original GPU-only liteFormer block in a HART/MobileHART repo copy.

This is used only for MobileHART calibration experiments. The upstream repo's
current `model.py` contains a CPU-compatible liteFormer rewrite; the file also
keeps the original NCW/GPU implementation in comments. This patch makes a
separate repo copy execute that original implementation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


OLD_LITEFORMER = r'''class liteFormer(layers.Layer):
    def __init__(self,startIndex,stopIndex, projectionSize, kernelSize = 16, attentionHead = 3, use_bias=False, dropPathRate = 0.0,dropout_rate = 0,**kwargs):
        super(liteFormer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.kernelSize = kernelSize
        self.softmax = tf.nn.softmax
        self.projectionSize = projectionSize
        self.attentionHead = attentionHead
        self.DropPathLayer = DropPath(dropPathRate)
        self.projectionHalf = projectionSize // 2

    def build(self,inputShape):
        self.depthwise_kernel = [self.add_weight(
            shape=(self.kernelSize,1,1),
            initializer="glorot_uniform",
            trainable=True,
            name="convWeights"+str(_),
            dtype="float32") for _ in range(self.attentionHead)]
        if self.use_bias:
            self.convBias = self.add_weight(
                shape=(self.attentionHead,),
                initializer="glorot_uniform",
                trainable=True,
                name="biasWeights",
                dtype="float32"
            )

    def call(self, inputs,training=None):
        formattedInputs = inputs[:,:,self.startIndex:self.stopIndex]
        inputShape = tf.shape(formattedInputs)
        reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,inputShape[1]))
        if(training):
            for convIndex in range(self.attentionHead):
                self.depthwise_kernel[convIndex].assign(self.softmax(self.depthwise_kernel[convIndex], axis=0))
        convOutputs = tf.convert_to_tensor([tf.nn.conv1d(
            reshapedInputs[:,convIndex:convIndex+1,:],
            self.depthwise_kernel[convIndex],
            stride = 1,
            padding = 'SAME',
            data_format='NCW',) for convIndex in range(self.attentionHead) ])
        convOutputsDropPath = self.DropPathLayer(convOutputs)
        localAttention = tf.reshape(convOutputsDropPath,(-1,inputShape[1],self.projectionSize))
        return localAttention

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_bias': self.use_bias,
            'kernelSize': self.kernelSize,
            'startIndex': self.startIndex,
            'stopIndex': self.stopIndex,
            'projectionSize': self.projectionSize,
            'attentionHead': self.attentionHead,})
        return config
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_py", type=Path)
    args = parser.parse_args()

    path = args.model_py
    text = path.read_text(encoding="utf-8")
    start_match = re.search(r"^class liteFormer\(layers\.Layer\):", text, re.MULTILINE)
    if start_match is None:
        raise SystemExit("could not find executable liteFormer class")
    start = start_match.start()
    end_match = re.search(r"^class mixAccGyro\(layers\.Layer\):", text, re.MULTILINE)
    if end_match is None:
        raise SystemExit("could not find mixAccGyro class")
    end = end_match.start()
    patched = text[:start] + OLD_LITEFORMER + "\n" + text[end:]
    path.write_text(patched, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
