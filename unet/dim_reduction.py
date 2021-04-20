import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import torch
import seaborn as sns
from model import UNet2D




# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


net = UNet2D(n_chans_in=1, n_chans_out=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
criterion, n_filters, batch_size, batches_per_epoch, n_epochs, lr_init, optimizer = net.default()


optimizer = optimizer(
        net.parameters(),
        lr=lr_init,
        momentum=0.9,
        nesterov=True,
        weight_decay=0
    )

model_path = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Test\checkpoints\CP_128_half.pth"

net.load_state_dict(torch.load(model_path, map_location=device))



net.conv1.register_forward_hook(get_activation('conv1'))

act = activation['conv1'].squeeze()
fig, axarr = plt.subplots(act.size(0))
for idx in range(act.size(0)):
    axarr[idx].imshow(act[idx])



N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.figure(figsize=(16,7))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)