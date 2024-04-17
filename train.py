
from unet import ContextUnet
from ddpm import DDPM
from data import Collator, Dataset

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from torch.utils.data import random_split, DataLoader
from datasets import load_dataset, concatenate_datasets


def train_mnist():

    # hardcoding these here
    n_epoch = 20
    batch_size = 256
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    # n_feat = 1024
    lrate = 1e-4
    save_model = False
    save_dir = './results/diffusion_outputs10/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5==0 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

def test():
    # hardcoding these here
    n_epoch = 20
    batch_size = 16
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    # n_feat = 1024
    lrate = 1e-4
    save_model = False
    save_dir = './results/diffusion_outputs10/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.5)
    ddpm.to(device)

    text_embs = torch.rand((batch_size, 8, n_feat)).to(device)
    c = torch.randint(0,9,(batch_size,)).to(device)
    x = torch.rand((batch_size,1,28,28)).to(device)
    loss = ddpm(x, c, text_embs)

def test_dataloader():
    image_label = None
    url_label = "link"
    text_label = "caption"
    text_encoder_name = "google/t5-v1_1-large"
    size = 64
    # dataset_name = "laion/laion2B-en"
    dataset_name = "laion/gpt4v-dataset"
    dataset_info = {"batch_size": 1, "shuffle": True}
    channels = "RGB"
    ds = load_dataset(dataset_name)
    
    train_ds = None
    
    # if we have train and valid split we combine them into one dataset to let trainer handle the split
    if 'train' in ds and 'valid' in ds:
        train_ds = concatenate_datasets([ds['train'], ds['valid']])
    elif 'train' in ds:
        train_ds = ds['train']
    elif 'valid' in ds:
        train_ds = ds['valid']
    else:
        train_ds = ds

    dl = make_train_dataset(
        ds = train_ds,
        collate_fn = Collator(
            image_size = size,
            image_label = image_label,
            text_label = text_label,
            url_label = url_label,
            name = text_encoder_name,
            channels = channels
        ),
        **dataset_info
    )
    for i, data in enumerate(dl):
        print(data)
        if i > 10:
            return


def make_train_dataset(ds = None, *, batch_size, **dl_kwargs):
        # if not exists(ds):
        #     return

        # assert not exists(self.train_dl), 'training dataloader was already added'

        # valid_ds = None
        # if self.split_valid_from_train:
        #     train_size = int((1 - self.split_valid_fraction) * len(ds))
        #     valid_size = len(ds) - train_size

        #     ds, valid_ds = random_split(ds, [train_size, valid_size], generator = torch.Generator().manual_seed(self.split_random_seed))
        #     self.print(f'training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples')

        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        # self.add_train_dataloader(dl)

        # if not self.split_valid_from_train:
        #     return

        # self.add_valid_dataset(valid_ds, batch_size = batch_size, **dl_kwargs)
        return dl




if __name__ == "__main__":
    # train_mnist()
    test_dataloader()
 