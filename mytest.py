import torch
import torchvision
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()
for p in model.parameters():
    p.requires_grad = False


class Block(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        def block(dim_in, dim_out, kernel_size=3, stride=1, padding=1):
            return (
                torch.nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
                torch.nn.BatchNorm2d(dim_out),
                torch.nn.LeakyReLU(),
            )

        self.s = torch.nn.Sequential(
            *block(dim_in, dim_in),
            *block(dim_in, dim_in),
            *block(dim_in, dim_in),
            *block(dim_in, dim_out, kernel_size=3, stride=2, padding=0),
            *block(dim_out, dim_out),
            *block(dim_out, dim_out),
            *block(dim_out, dim_out),
        )

        self.res = torch.nn.ConvTranspose2d(dim_in, dim_out, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.s(x) + self.res(x)


gen = torch.nn.Sequential(
    torch.nn.Linear(128, 256 * 4 * 4),
    torch.nn.InstanceNorm1d(256 * 4 * 4),
    torch.nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
    Block(256, 128),
    Block(128, 64),
    Block(64, 32),
    Block(32, 3),
    torch.nn.UpsamplingNearest2d(size=(64, 64)),
    torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
    torch.nn.Tanh(),
)
gen.to(device)
a = torch.randn((1, 128)).to(device)
optimizer = torch.optim.Adam(gen.parameters(), lr=0.01)

my_trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224), antialias=True),
    torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# image = gen(a).squeeze()
image_test2 = Image.open("/data/jbwang/nerf_test_data/lego/train/r_0.png")
image_test = torch.from_numpy(np.array(image_test2)).permute(2, 0, 1).float() / 255
image_test2 = image_test[:3] * image_test[-1:] + (1. - image_test[-1:])
# plt.figure(figsize=(8, 8))
# # plt.imshow(image_test.permute(1, 2, 0).detach().cpu().numpy())
# plt.imshow(image_test.permute(1, 2, 0))
# plt.axis('off')
# plt.show()

for _ in range(500):
    # image = gen(a).squeeze()
    # b_image = Image.fromarray(image.permute(1, 2, 0).byte().cpu().numpy())
    # c_image = preprocess(b_image).unsqueeze(0).to(device)

    # image_test = preprocess(image_test2).unsqueeze(0).to(device)
    # c_image = my_trans(image).unsqueeze(0).to(device)
    image_test = my_trans(image_test2).unsqueeze(0).to(device)

    text = clip.tokenize(["a large blue bird standing next to a painting of flowers"]).to(device)
    image_features = model.encode_image(image_test)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    l = -(image_features * text_features).sum(-1).mean()
    print(l)
    # optimizer.zero_grad()
    # l.backward()
    # optimizer.step()
    # print(l)
