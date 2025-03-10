import torch
from . import initialization as init
import torch.nn as nn
import torch.nn.functional as F



class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head_1)
        init.initialize_head(self.segmentation_head_3)
        init.initialize_head(self.segmentation_head_7)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output1, decoder_output3, decoder_output7 = self.decoder(*features)
        
        masks1 = self.segmentation_head_1(decoder_output1)
        masks3 = self.segmentation_head_3(decoder_output3)
        masks7 = self.segmentation_head_7(decoder_output7)
        x=torch.concat((masks1,masks3,masks7), 1)
        x=self.fusion(x)

        return [masks1,x,masks3,masks7]

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
