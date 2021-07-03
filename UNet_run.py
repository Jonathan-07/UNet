""" Load model """
import torch
from PIL import Image
from torchvision.utils import save_image
import pandas as pd


from UNet_full_network import *
from datainporter import *




hyperparams = 'lr0.1_bs1_mom0.99_epoch5000'
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load('UNet_{}.pth'.format(hyperparams)))
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = model.to(device)

ism_folder = 'ISM_Val'
conf_folder = 'Conf_Val'
tile_width_split = 3
tile_height_split = 6
tilesplit = tile_height_split * tile_width_split
dataset = superdata(conf_folder, ism_folder, tile_height_split, tile_width_split)
loader = DataLoader(dataset, batch_size= 2, shuffle=False)

# dataset_size = dataset.__len__()
# dict = {i:[] for i in range(int(dataset_size/tilesplit))}

acc_list = []

i=0
with torch.no_grad():
    for batch in loader:

        images, masks = batch
        images = images.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if network.n_classes == 1 else torch.long
        masks = masks.to(device=device, dtype=mask_type)
        images = images.unsqueeze(1)
        masks = masks.unsqueeze(1)

        output = network(images)


        # if network.n_classes > 1:
        #     output = F.softmax(output, dim=1)
        # else:
        #     output = torch.sigmoid(output)

        # output = output.squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(full_img.size[1]),
        #         transforms.ToTensor()
        #     ]
        # )
        # unloader = transforms.ToPILImage()
        # probs = probs.cpu()

        output = output.detach()
        output_tensor = output.squeeze().cpu()
        output = output.squeeze().cpu().numpy()
        masks = masks.cpu()
        for x in range(loader.batch_size):
            y = i*loader.batch_size + x
            if y > 2:
                break

            print(y)

            if y == 2:
                accu = accuracy(output_tensor, masks, 0.05)[0]
                acc_list.append(accu)
                acc_arr = accuracy(output_tensor, masks, 0.05, need_array=True)[1]
                im = Image.fromarray(output, mode='F')  # float32

            else:
                accu = accuracy(output_tensor[x], masks[x], 0.05)[0]
                acc_list.append(accu)
                acc_arr = accuracy(output_tensor[x], masks[x], 0.05, need_array=True)[1]
                im = Image.fromarray(output[x], mode='F')  # float32

            if y == 0:
                tag = 12
            elif y == 1:
                tag = 25
            elif y == 2:
                tag = 38

            im.save('./output/5000 epochs/{}_super-res_{}.tiff'.format(tag, hyperparams), "TIFF")
            np.save('./output/5000 epochs/acc{}_{}'.format(tag, hyperparams), acc_arr)

        # output_im = unloader(output)

        # plt.figure()
        # plt.imshow(output[0, :, :])

        # output_im = Image.fromarray((output).astype(np.uint8))

        # plt.subplot(1,2,1)
        # plt.imshow(output_im)
        # plt.subplot(1,2,2)
        # plt.imshow(masks[0,:,:])
        # plt.show()

        # x = (i - 1) // tilesplit
        # dict[x].append(output)
        # if 1 <= i <= 18:
        #     B.append(output_im)
        # elif 19 <= i <= 36:
        #     G.append(output_im)
        # elif 37 <= i <= 54:
        #     R.append(output_im)
        i += 1

df = pd.DataFrame(acc_list, columns=['Accuracy'])
df.to_csv('./output/5000 epochs/Accuracy_df_{}'.format(hyperparams))

# def merge_images(list):
#     (width, height) = list[0].size
#     result_width = width * 3
#     result_height = height * 6
#
#     result = Image.new('L', (result_width, result_height))
#     for j in range(len(list)):
#         if 0 <= j <= 5:
#             result.paste(im = list[j], box = (0, height*j))
#         elif 6 <= j <= 11:
#             result.paste(im = list[j], box = (width, height*(j-6)))
#         elif 12 <= j <= 17:
#             result.paste(im = list[j], box = (width*2, height*(j-12)))
#
#     return result

def merge_arrays(arr_list, width_split, height_split):
    dict_merge = {k:[] for k in range(width_split)}
    for k in range(width_split):
        dict_merge[k] = np.vstack(arr_list[j] for j in range(height_split * k, height_split * (k+1)))

    result = np.hstack( dict_merge[x] for x in range(width_split))
    # result = ((result - np.min(result)) / (np.max(result) - np.min(result)))

    return result



# B_final = merge_images(B)
# G_final = merge_images(G)
# R_final = merge_images(R)

# for i in range(int(dataset_size/tilesplit)):
#     dict[i] = merge_images(dict[i])
#     dict[i].save('./output/{}_super-res_MSE.png'.format(i))
#     print(i)

# for i in range(int(dataset_size/tilesplit)):
#     dict[i] = merge_arrays(dict[i], tile_width_split, tile_height_split)
#     im = Image.fromarray(dict[i], mode='F')  # float32
#     im.save('./output/{}_super-res_MSE.tiff'.format(i), "TIFF")
#     # dict[i].save('./output/{}_super-res_MSE.png'.format(i))
#     print(i)

# save_image(B_final, './output/B_super-res.tif')
# save_image(G_final, './output/G_super-res.tif')
# save_image(R_final, './output/R_super-res.tif')

# B_final.save('./output/B_super-res_MSE.png')
# G_final.save('./output/G_super-res_MSE.png')
# R_final.save('./output/R_super-res_MSE.png')

# plt.imshow(B_final)
# plt.show()
# plt.imshow(G_final)
# plt.show()
# plt.imshow(R_final)
# plt.show()
