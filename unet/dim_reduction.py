import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import sklearn.manifold
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import models
from model import UNet2D
from misc_functions import preprocess_image, recreate_image, save_image
from PIL import Image
from utils.dataset import BasicDataset
class CNNLayerVisualization():

    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """

    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(0, 255, (128, 128, 1)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = torch.optim.SGD(
                net.parameters(),
                lr=0.001,
                momentum=0.9,
                nesterov=True,
                weight_decay=0
            )
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            full_img = Image.open(os.path.join(os.getcwd(),'data','input_s.jpg'))
            full_img = np.array(full_img)
            img = torch.from_numpy(BasicDataset.preprocess(full_img, (128,128)))

            x = img.unsqueeze(0)
            x = x[:, 0, :, :].reshape(1, 1, 128, 128)
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(x)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                          '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook
def remove_far(input):

    mean = np.absolute(np.mean(input, axis=0)).sum()
    sd = np.absolute(np.std(input, axis=0)).sum()

    final_list = [x for x in input if (np.absolute(x).sum() > mean - 2 * sd)]
    final_list = [x for x in final_list if (np.absolute(x).sum() < mean + 2 * sd)]
    return np.array((final_list))

def get_mid_output(net, fig, img,col):
    # img = Image.open(os.path.join(os.getcwd(), 'data', 'input_s.jpg'))
    net.down2.register_forward_hook(get_activation('down2'))
    net.eval()
    full_img = np.array(img)
    img = torch.from_numpy(BasicDataset.preprocess(full_img, (128, 128)))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        output_mid = activation['down2'].cpu().detach().numpy()[0]
        output_mid = output_mid.reshape(32, -1)
        output_2d = sklearn.manifold.TSNE(n_components=3, perplexity=5).fit_transform(output_mid)
        # output_2d = remove_far(output_2d)
        # plt.scatter(output_2d[:,0],output_2d[:,1],c=col)
        ax.scatter(*zip(*output_2d), c=col)
if __name__ == '__main__':


    fig3d = plt.figure()
    ax = fig3d.add_subplot(111, projection='3d')

    cnn_layer = 1
    filter_pos = 2
    # Fully connected layer is not needed
    net = UNet2D(n_chans_in=1, n_chans_out=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device='cuda')

    model_path = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Test\checkpoints\CP_epoch79.pth"
    net.load_state_dict(torch.load(model_path, map_location=device))

    #fig = plt.figure()

    img = Image.open(os.path.join(os.getcwd(), 'data', 'input_p_1.jpg'))

    get_mid_output(net, ax, img, 'r')
    img = Image.open(os.path.join(os.getcwd(), 'data', 'input_p_2.jpg'))

    get_mid_output(net, ax, img,'r')
    img = Image.open(os.path.join(os.getcwd(), 'data', 'input_p_3.jpg'))

    get_mid_output(net, ax, img,'r')
    img = Image.open(os.path.join(os.getcwd(), 'data', 'input_p_4.jpg'))

    get_mid_output(net, fig3d, img,'r')
    img = Image.open(os.path.join(os.getcwd(), 'data', 'input_p_5.jpg'))

    get_mid_output(net, ax, img,'r')
   # plt.show()
    img = Image.open(os.path.join(os.getcwd(), 'data', 'mask.jpg'))

    get_mid_output(net, fig3d, img,'b')
    plt.show()

    # layer_vis = CNNLayerVisualization(net, cnn_layer, filter_pos)
    #
    # # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_with_hooks()
    #
    # # Layer visualization without pytorch hooks
    # # layer_vis.visualise_layer_without_hooks()