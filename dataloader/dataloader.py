from torch.utils.data import Dataset, DataLoader
import torch
from utils.util import *
import pandas as pd


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None, CFG=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.tagging = df['tagging'].tolist()
        self.transforms = transforms
        self.CFG = CFG

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path, self.CFG)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            timestamp = self.tagging[index]
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk), torch.tensor(timestamp)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


def prepare_loaders(df_test, df, fold, CFG, debug=False, transforms=None):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    test_df = df_test

    if debug:
        train_df = train_df.head(32 * 5).query("empty==0")
        valid_df = valid_df.head(32 * 3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=transforms['train'], CFG=CFG)
    valid_dataset = BuildDataset(valid_df, transforms=transforms['valid'], CFG=CFG)
    test_dataset = BuildDataset(test_df, transforms=transforms['valid'], CFG=CFG)
    # distribution = get_distribution(train_dataset)
    # distribution.to_csv('/home8t/rgye/seg_rgye/distribution.csv')

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=CFG['train_bs'] if not debug else CFG['debug_train_bs'],
                              num_workers=4, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, shuffle=True,
                              batch_size=CFG['valid_bs'] if not debug else CFG['debug_valid_bs'],
                              num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=CFG['valid_bs'] if not debug else CFG['debug_valid_bs'],
                             num_workers=4, pin_memory=True, drop_last=False)

    # return train_sampler, valid_sampler, train_loader, valid_loader
    return train_loader, valid_loader, test_loader

# def get_distribution(dataset):
#     contrast_matrix = torch.zeros(dataset[0][1][0].shape)
#
#     distribution = pd.DataFrame()
#     timestamp = []
#     channel_0 = []
#     channel_1 = []
#     channel_2 = []
#     for i in range(len(dataset)):
#         timestamp.append(dataset[i][2].item())
#         if torch.equal(contrast_matrix, dataset[i][1][0]):
#             # distribution.append([{'channel_0': 0}], ignore_index=True)
#             channel_0.append(0)
#         else:
#             # distribution.append([{'channel_0': 1}], ignore_index=True)
#             channel_0.append(1)
#         if torch.equal(contrast_matrix, dataset[i][1][1]):
#             # distribution.append([{'channel_1': 0}], ignore_index=True)
#             channel_1.append(0)
#         else:
#             # distribution.append([{'channel_1': 1}], ignore_index=True)
#             channel_1.append(1)
#         if torch.equal(contrast_matrix, dataset[i][1][2]):
#             # distribution.append([{'channel_2': 0}], ignore_index=True)
#             channel_2.append(0)
#         else:
#             # distribution.append([{'channel_2': 1}], ignore_index=True)
#             channel_2.append(1)
#
#         print('*' * 25)
#         print('{:.4f}'.format(i/len(dataset)))
#
#     distribution['timestamp'] = timestamp
#     distribution['channel_0'] = channel_0
#     distribution['channel_1'] = channel_1
#     distribution['channel_2'] = channel_2


# return distribution
