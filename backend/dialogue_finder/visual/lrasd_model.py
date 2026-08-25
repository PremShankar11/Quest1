# MIT License
#
# Copyright (c) 2025 Liao Junhua
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Vendored active-speaker-detection network from Junhua-Liao/LR-ASD (MIT).

Upstream repository: https://github.com/Junhua-Liao/LR-ASD
Upstream commit:      1b6dcd2d8fc2895683de6508ec6294ec47d388ca (2025-03-23)
Vendored from:        model/Model.py, model/Encoder.py, model/Classifier.py

What changed from upstream:
  - `ASD_Model` (Model.py), `visual_encoder`/`audio_encoder` (Encoder.py) and
    `Fusion`/`Detector` (Classifier.py) are copied verbatim (module structure,
    forward signatures, weight-init) so the pretrained state dict loads
    unmodified.
  - The training wrapper `ASD.py` (optimizer, scheduler, `.cuda()` calls,
    `train_network`/`evaluate_network`, `lossAV`/`lossV` BCELoss+accuracy
    machinery) is NOT vendored — no training code, CPU only.
  - `lossAV.FC` (`nn.Linear(128, 2)`), the classification head baked into the
    same checkpoint, is reproduced here as `AVScoreHead` (forward only, no
    loss). This is required for inference: `ASD_Model.forward` stops at the
    128-d fused embedding.
  - `lossV.FC` (visual-only auxiliary head, training-only) is NOT vendored.

No code below executes at import time beyond class/function definitions —
`torch` is imported at module level here because this file is only ever
imported lazily by later, non-vendored code (`visual/lrasd.py`), never from a
module-level import in pipeline code.

Checkpoint key mapping (see docs/superpowers/spikes/2026-08-25-lrasd-spike.md,
`weights` / `model_api` headings for the exact file names and shapes):
  state_dict["model.<name>"]    -> ASD_Model()'s own state_dict()["<name>"]
  state_dict["lossAV.FC.weight"] -> AVScoreHead()'s state_dict()["FC.weight"]
  state_dict["lossAV.FC.bias"]   -> AVScoreHead()'s state_dict()["FC.bias"]
  (i.e. strip the "lossAV." prefix, keep "FC.")
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2):
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1 - 1) / 2)
        self.padding_2 = int((kernel_2 - 1) / 2)

        self.m_1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(kernel_1, 1), padding=(self.padding_1, 0), bias=False)
        self.m_norm_1 = nn.BatchNorm2d(out_channels // 2, momentum=0.01, eps=0.001)
        self.m_2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=(kernel_2, 1), padding=(self.padding_2, 0), bias=False)
        self.m_norm_2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.t_1 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_1), padding=(0, self.padding_1), bias=False)
        self.t_norm_1 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_2), padding=(0, self.padding_2), bias=False)
        self.t_norm_2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.m_norm_1(self.m_1(x)))
        x = self.relu(self.m_norm_2(self.m_2(x)))

        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))

        return x


class Visual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down=False):
        super(Visual_Block, self).__init__()

        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1 - 1) / 2)
        self.padding_2 = int((kernel_2 - 1) / 2)

        if is_down:
            self.s_1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=(1, kernel_1, kernel_1), stride=(1, 2, 2), padding=(0, self.padding_1, self.padding_1), bias=False)
        else:
            self.s_1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=(1, kernel_1, kernel_1), padding=(0, self.padding_1, self.padding_1), bias=False)

        self.s_norm_1 = nn.BatchNorm3d(out_channels // 2, momentum=0.01, eps=0.001)

        self.s_2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=(1, kernel_2, kernel_2), padding=(0, self.padding_2, self.padding_2), bias=False)
        self.s_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.t_1 = nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_1, 1, 1), padding=(self.padding_1, 0, 0), bias=False)
        self.t_norm_1 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_2 = nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_2, 1, 1), padding=(self.padding_2, 0, 0), bias=False)
        self.t_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.s_norm_1(self.s_1(x)))
        x = self.relu(self.s_norm_2(self.s_2(x)))

        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))

        return x


class visual_encoder(nn.Module):
    def __init__(self):
        super(visual_encoder, self).__init__()

        self.block1 = Visual_Block(1, 32, 5, 3, is_down=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.block2 = Visual_Block(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.block3 = Visual_Block(64, 128, 5, 3)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = x.transpose(1, 2)
        B, T, C, W, H = x.shape
        x = x.reshape(B * T, C, W, H)

        x = self.maxpool(x)

        x = x.view(B, T, C)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    def __init__(self):
        super(audio_encoder, self).__init__()

        self.block1 = Audio_Block(1, 32, 5, 3)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))

        self.block2 = Audio_Block(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))

        self.block3 = Audio_Block(64, 128, 5, 3)

        self.__init_weight()

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim=2, keepdim=True)
        x = x.squeeze(2).transpose(1, 2)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Conv1d(channel, channel, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(channel, momentum=0.01, eps=0.001)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 2)
        identity = x.transpose(1, 2)
        w = self.sigmoid(self.bn(self.attention(identity)))
        x = (identity * w).transpose(1, 2)
        return x


class Detector(nn.Module):
    def __init__(self, channel):
        super(Detector, self).__init__()

        self.gru_forward = nn.GRU(input_size=channel, hidden_size=channel // 4, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.gru_backward = nn.GRU(input_size=channel, hidden_size=channel // 4, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.attention = Fusion(channel // 2)
        self.__init_weight()

    def forward(self, x):
        x1, _ = self.gru_forward(self.drop(x))
        x = torch.flip(x, dims=[1])
        x2, _ = self.gru_backward(self.drop(x))
        x2 = torch.flip(x2, dims=[1])
        x = self.attention(x1, x2)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()


class ASD_Model(nn.Module):
    """Backbone: fused audio+visual embedding, no classification head.

    forward(audioFeature, visualFeature) -> (outsAV, outsV), each (B*T, 128).
      audioFeature:  float tensor (B, T_a, 13)      T_a = 100 Hz MFCC frame count
      visualFeature: float tensor (B, T_v, 112, 112) uint8-range grey crops (0..255)
      constraint:    T_a must equal 4 * T_v exactly (Fusion concatenates along
                     the channel axis after the audio path's 2x stride-2 time
                     pools bring T_a down to T_v).
    outsAV is the fused audio-visual embedding (what AVScoreHead consumes);
    outsV is a visual-only embedding used upstream only for the auxiliary
    lossV during training (not needed for inference).
    """

    def __init__(self):
        super(ASD_Model, self).__init__()

        self.visualEncoder = visual_encoder()
        self.audioEncoder = audio_encoder()
        self.fusion = Fusion(256)
        self.detector = Detector(256)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):
        x = self.fusion(x1, x2)
        x = self.detector(x)
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self, x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward(self, audioFeature, visualFeature):
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        outsV = self.forward_visual_backend(visualEmbed)

        return outsAV, outsV


class AVScoreHead(nn.Module):
    """Classification head on top of ASD_Model's fused embedding.

    Not present in upstream Model.py — upstream bundles this Linear(128, 2)
    inside the training-only `lossAV` module (loss.py), alongside a BCELoss
    that this vendor drops. Loads from checkpoint keys `lossAV.FC.weight`
    (2, 128) and `lossAV.FC.bias` (2,).

    forward(x) -> raw 2-class logits, shape (N, 2).
    Probability that the track is the active speaker at each of the N frames:
        prob = softmax(head(outsAV), dim=-1)[:, 1]
    (NOT `logits[:, 1] >= 0`; that is the upstream demo script's own
    visualization shortcut, a different and looser decision rule than a
    calibrated 0.5 probability threshold — see the spike note.)
    """

    def __init__(self):
        super(AVScoreHead, self).__init__()
        self.FC = nn.Linear(128, 2)

    def forward(self, x):
        return self.FC(x)
