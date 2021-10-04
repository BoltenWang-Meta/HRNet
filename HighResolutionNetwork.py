import torch

from    torch import nn

class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, fz=3, step=1, act=False):
        super(Conv, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=step),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, kernel_size=fz, stride=step, padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=step),
            nn.BatchNorm2d(ch_out),
        )
        if act:
            self.relu = nn.ReLU()
        self.act = act
        self.ch_in = ch_in
    def forward(self, inport):
        assert inport.shape[1] == self.ch_in
        out = self.blk(inport)
        if self.act:
            out = self.relu(out)
        return out

class BasicBlk(nn.Module):
    def __init__(self, ch_in):
        super(BasicBlk, self).__init__()
        self.ch_in = ch_in
        blk = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_in),
        )

        self.conv = nn.ModuleList([blk for _ in range(4)])
        self.act = nn.ModuleList([nn.ReLU() for _ in range(4)])
    def forward(self, inport):
        assert inport.shape[1] == self.ch_in
        out = inport
        for layer_num in range(4):
            out_conv = self.conv[layer_num](out)
            out = self.act[layer_num](out + out_conv)
        return out

class BottleNeck(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(BottleNeck, self).__init__()
        self.ch_in = ch_in
        self.conv_1 = Conv(ch_in, ch_out)

        self.bottle = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out)
        )
        self.relu = nn.ReLU()

        self.conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch_out, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, ch_out, kernel_size=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(),
            ) for _ in range(3)
        ])
    def forward(self, inport):
        assert inport.shape[1] == self.ch_in
        out = self.conv_1(inport)

        out_b = self.bottle(inport)
        out = self.relu(out+out_b)
        for layer in self.conv_2:
            out = layer(out)
        return out

class transition(nn.Module):
    def __init__(self, out_branch):
        super(transition, self).__init__()
        self.out_branch = out_branch
        if out_branch != 2:
            encode = [32, 64, 128, 256]
            encode_out = [32, 64, 128, 256]
        else:
            encode = [256, 256]
            encode_out = [32, 64, 128, 256]
        self.conv = nn.ModuleList([])
        for branch in range(out_branch-1):
            self.conv.append(nn.Conv2d(encode[branch], encode_out[branch], kernel_size=3, padding=1, stride=1))

        self.conv.append(nn.Conv2d(encode[out_branch-2], encode_out[out_branch-1], kernel_size=3, padding=1, stride=2))

    def forward(self, inports):
        print(len(inports), self.out_branch-1)
        assert len(inports) == self.out_branch-1
        res = []
        for bra in range(len(inports)):
            res.append(self.conv[bra](inports[bra]))
        res.append(self.conv[-1](inports[-1]))
        return res

class Fuze(nn.Module):
    def __init__(self, ori_bra, tar_bra):
        super(Fuze, self).__init__()
        delta = tar_bra - ori_bra
        encode = [32, 64, 128, 256]
        if delta > 0:
            self.blk = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(encode[ori_bra], encode[ori_bra], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(encode[ori_bra]), nn.ReLU()
                ) for _ in range(delta-1)
            ])
            self.blk.append(
                nn.Sequential(
                    nn.Conv2d(encode[ori_bra], encode[tar_bra], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(encode[tar_bra])
                )
            )
        elif delta < 0:
            self.blk = nn.ModuleList([nn.Sequential(
                nn.Conv2d(encode[ori_bra], encode[tar_bra], kernel_size=1),
                nn.Upsample(scale_factor=2 ** (-delta), mode='nearest'),
                nn.BatchNorm2d(encode[tar_bra])
            )])
    def forward(self, inport):
        out = inport
        for layer in self.blk:
            out = layer(out)
        return out


class HRModule(nn.Module):
    def __init__(self, num_bra):
        super(HRModule, self).__init__()
        self.bra = num_bra
        encode = [32, 64, 128, 256]
        self.BasicBlk = nn.ModuleList([])
        self.Fuzes = nn.ModuleList([])
        for bra in range(num_bra):
            self.BasicBlk.append(BasicBlk(encode[bra]))
            self.Fuzes.append(nn.ModuleList([]))
            for tar in range(num_bra):
                if bra != tar:
                    self.Fuzes[-1].append(Fuze(bra, tar))
                else:
                    self.Fuzes[-1].append(None)
        self.act = nn.ModuleList([nn.ReLU() for _ in range(num_bra)])

    def forward(self, inport):
        assert len(inport) == self.bra
        res = [0 for _ in range(len(inport))]
        for bra in range(self.bra):
            out_Basic = self.BasicBlk[bra](inport[bra])
            for tar in range(self.bra):
                if tar != bra and type(res[tar]) != int:
                    target = self.Fuzes[bra][tar](out_Basic)

                    res[tar] = res[tar] + target
                elif tar != bra and type(res[tar]) == int:
                    target = self.Fuzes[bra][tar](out_Basic)

                    res[tar] = target
        for num in range(len(res)):
            res[num] = self.act[num](res[num])
        return res



class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.stage1 = BottleNeck(64, 256)
        self.trans1 = transition(2)

        self.stage2 = HRModule(2)
        self.trans2 = transition(3)

        self.stage3 = nn.ModuleList([
            HRModule(3) for _ in range(4)
        ])

        self.trans3 = transition(4)

        self.stage4 =  nn.ModuleList([
            HRModule(4) for _ in range(2)
        ])


    def forward(self, img):
        print('########## in stem ##########')
        out_stem = self.stem(img)
        print('########## out stem ##########', out_stem.shape)

        print('########## in stage_1 ##########')
        out_stage_1 = self.stage1(out_stem)
        print('########## out stage_1 ##########', out_stage_1.shape)

        print('########## in trans1 ##########')
        out_trans1 = self.trans1(out_stage_1.unsqueeze(0))
        print('########## out trans1 ##########', out_trans1[0].shape, out_trans1[1].shape)

        print('########## in stage_2 ##########')
        out_stage_2 = self.stage2(out_trans1)
        print('########## out stage_2 ##########', out_stage_2[0].shape, out_stage_2[1].shape)

        print('########## in trans2 ##########')
        out_trans2 = self.trans2(out_stage_2)
        print('########## out trans2 ##########', out_trans2[0].shape, out_trans2[1].shape, out_trans2[2].shape)

        print('########## in stage_3 ##########')
        out_stage_3 = out_trans2
        for layer in self.stage3:
            out_stage_3 = layer(out_stage_3)
        print('########## out stage_3 ##########', out_stage_3[0].shape, out_stage_3[1].shape, out_stage_3[2].shape)

        print('########## in trans3 ##########')
        out_trans3 = self.trans3(out_stage_3)
        print('########## out trans3 ##########', out_trans3[0].shape, out_trans3[1].shape, out_trans3[2].shape, out_trans3[3].shape)

        print('########## in stage_4 ##########')
        out_stage_4 = out_trans3
        for layer in self.stage4:
            out_stage_4 = layer(out_stage_4)
        print('########## out stage_4 ##########', out_stage_4[0].shape, out_stage_4[1].shape, out_stage_4[2].shape, out_stage_4[3].shape)

        out = out_stem
        return out

def test():
    test_map = torch.rand((1, 3, 256, 256))
    model = HRNet()
    out = model(test_map)
    print('\n', out.shape)


if __name__ == "__main__":
    test()
