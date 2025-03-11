import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(1,16,3,1,1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.con1_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con2_1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con3_1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.con4_1 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # self.att1 = SA(32)
        # self.att2 = SA(64)
        # self.att3 = SA(128)
        self.decon1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decon2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(), )

        self.decon3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),)

        self.decon4 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
        )

        self.cov1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.ReLU()
        )
        self.ca1 = SA(64)
        self.ca2 = SA(32)
        self.ca3 = SA(16)

    def forward(self,x,y):
        E1_1 = self.con1(x)
        E1_2 = self.con1_1(y)

        E2_1 = self.con2(E1_1)
        E2_2 = self.con2_1(E1_2)

        E3_1 = self.con3(torch.cat([E2_1,E1_1],1))
        E3_2 = self.con3_1(torch.cat([E2_2,E1_2],1))

        E4_1 = self.con4(torch.cat([E3_1,E2_1,E1_1],1))
        E4_2 = self.con4_1(torch.cat([E3_2,E2_2,E1_2],1))

        Fc = torch.cat([ E4_1,E3_1,E2_1,E1_1,E4_2, E3_2,E2_2,E1_2],1)
        FA = torch.cat([E4_1,E3_1,E2_1,E1_1],1)
        FB = torch.cat([E4_2, E3_2,E2_2,E1_2], 1)


        Df1 = self.decon1(Fc)
        ca1 = self.ca1(Df1)*(self.cov1(FA.detach()) + self.cov1(FB.detach()))
        Df2 = self.decon2(Df1+ca1)
        ca2 =  self.ca2(Df2)*(self.cov2(FA.detach()) + self.cov2(FB.detach()))
        Df3 = self.decon3(Df2+ca2)
        ca3 =  self.ca3(Df3)*(self.cov3(FA.detach()) + self.cov3(FB.detach()))
        F = self.decon4(Df3+ca3)


        return [E1_1,E1_2],[E2_1,E2_2],[E3_1,E3_2],[E4_1,E4_2],Fc,F

class intr(nn.Module):
    def __init__(self):
        super(intr, self).__init__()

    def forward(self,f_cat):
        (b, c, h, w) = f_cat.size()
        fr = f_cat.reshape(b,c, -1)
        fr_T = fr.transpose(1,2)
        W = torch.softmax(torch.matmul(fr,fr_T), 1)
        fw = torch.matmul(W,fr)
        f = fw.reshape(b, c, h, w)


        return norm_1((f_cat - torch.mean(f_cat)).clamp_min(1e-10))+f+f_cat

class SA(nn.Module):
    def __init__(self,inchan):
        super(SA, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.cov = nn.Sequential(
            nn.Conv2d(inchan*2,1,3,1,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        sa1 = self.cov(torch.cat([self.max(x),self.avg(x)],1))
        # sa2 = self.cov(torch.cat([self.max(y), self.avg(y)], 1))
        return sa1



class Fusion_strage(nn.Module):
    def __init__(self,in_channel=64):
        super(Fusion_strage,self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channel*2,in_channel,3,1,1),
            nn.LeakyReLU()
        )
        self.activation = nn.Sequential(
            nn.Conv2d(in_channel, 1, 3, 1, 1),

        )
        self.Mm = intr()

    def forward(self,E1,E2,E3,E4,fc):

        cat_features= self.conv(fc)
        # N, C, H, W = cat_features.size()

        activation_maps = self.activation(cat_features)  #map
        # norm_activation_maps = torch.split(activation_maps,1,1)
        # norm_activation_maps = torch.split(activation_maps,1,1)
        # activation_maps1 = np.asarray(activation_maps.data.cpu())
        # norm_activation_maps = nn.Sigmoid()(activation_maps)
        # norm_activation_maps = torch.split(activation_maps,1,1)
        # norm_activation_maps2 = norm_1(norm_activation_maps[1])
        # norm_activation_maps1 = norm_1(norm_activation_maps[0])
        # Ir_activation = norm_activation_maps1
        # Vi_activation = norm_activation_maps2

        norm_activation_maps1 = norm_1(activation_maps)
        Ir_activation = norm_activation_maps1
        Vi_activation = 1 - norm_activation_maps1

        ME1_1 ,ME1_2 = self.Mm(E1[0]),self.Mm(E1[1])
        ME2_1, ME2_2 = self.Mm(E2[0]), self.Mm(E2[1])
        ME3_1, ME3_2 = self.Mm(E3[0]), self.Mm(E3[1])
        ME4_1, ME4_2 = self.Mm(E4[0]), self.Mm(E4[1])

        M1 = torch.cat([ME1_1,ME2_1,ME3_1,ME4_1],1)
        M2 = torch.cat([ME1_2, ME2_2, ME3_2, ME4_2], 1)

        Ir_activation_map = Ir_activation * cat_features
        Vi_activation_map = Vi_activation * cat_features

        # return [ME1_1.reshape(E4[0].size(0), -1),  ME1_2.reshape(E4[0].size(0), -1)],[ME2_1.reshape(E4[0].size(0), -1),  ME2_2.reshape(E4[0].size(0), -1)],\
        #        [ME3_1.reshape(E4[0].size(0), -1),  ME3_2.reshape(E4[0].size(0), -1)],[ME4_1.reshape(E4[0].size(0), -1),  ME4_2.reshape(E4[0].size(0), -1)],\
        #        [Ir_activation_map.reshape(E4[0].size(0), -1),  Vi_activation_map.reshape(E4[0].size(0), -1)], [Ir_activation,Vi_activation]
        return [M1,M2],\
               [M1.reshape(E4[0].size(0), -1),M2.reshape(E4[0].size(0), -1)],\
               [Ir_activation_map.reshape(Ir_activation_map.size(0), -1),  Vi_activation_map.reshape(Vi_activation_map.size(0), -1)],\
               [Ir_activation,Vi_activation]


def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    return (x - min1) / (max1 - min1 + 1e-10)