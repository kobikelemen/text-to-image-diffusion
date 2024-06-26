
import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_EMB_DIM = 1024
# TEXT_EMB_DIM = 10


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [
            ResidualConvBlock(in_channels, out_channels), 
            # attn applied here (for now only in UnetUp)
            nn.MaxPool2d(2),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.cross_attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=1, batch_first=True, kdim=out_channels, vdim=out_channels) # TODO are these params correct?
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip, text_embs):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        x_shape = x.shape
        x_reshape = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        x_reshape = x_reshape.permute(0,2,1) 
        k = torch.cat((x_reshape,text_embs), dim=1)
        v = torch.cat((x_reshape,text_embs), dim=1)
        x, _ = self.cross_attn(query=x_reshape, key=k, value=v)
        return x.permute(0,2,1).view(x_shape)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)
        self.textembed1 = EmbedFC(TEXT_EMB_DIM, 2*n_feat)
        self.textembed2 = EmbedFC(TEXT_EMB_DIM, 1*n_feat)
        self.textatt1 = EmbedFC(TEXT_EMB_DIM, 1*n_feat)
        self.textatt2 = EmbedFC(TEXT_EMB_DIM, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
    
    def avg_pool(self, text_embs):
        return torch.mean(text_embs.float(), dim=1)

    def forward(self, x, c, t, context_mask, context_mask_text, text_embs):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        
        pool_text_emb = self.avg_pool(text_embs)

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask

        # mask out text_embs if context_mask_text == 1
        context_mask_text = (-1*(1-context_mask_text)) # need to flip 0 <-> 1
        text_embs = text_embs * context_mask_text
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        textembed1 = self.textembed1(pool_text_emb).view(-1, self.n_feat * 2, 1, 1)
        textembed2 = self.textembed2(pool_text_emb).view(-1, self.n_feat, 1, 1)
        
        text_embs_reshape = text_embs.view(-1,TEXT_EMB_DIM)
        textatt1 = self.textatt1(text_embs_reshape).view(text_embs.shape[0],text_embs.shape[1],-1)
        textatt2 = self.textatt2(text_embs_reshape).view(text_embs.shape[0],text_embs.shape[1],-1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)
        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1 + textembed1, down2, textatt1)
        up3 = self.up2(up2 + temb2 + textembed2, down1, textatt2)
        out = self.out(torch.cat((up3, x), 1))
        return out
