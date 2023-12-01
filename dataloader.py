import numpy as np
from torch.utils.data import Dataset
import torch


def data_load(data_name):
    data_dir = "./data/"
    data_path = data_dir + data_name
    f = open(data_path)  # 读取文件
    lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip().split("\t")[1:]

    gene_expression = np.zeros([len(lines) - 1, len(lines[1])])

    for i in range(len(lines)):
        if i == 0:
            continue
        for j in range(len(lines[1])):
            gene_expression[i - 1][j] = float(lines[i][j])

    gene_expression = np.matrix(gene_expression)
    gene_expression = np.transpose(gene_expression)
    gene_expression = np.array(gene_expression)

    return gene_expression


def load_data1(data_name):
    gene_expression = data_load(data_name)
    Label = gene_expression
    Img = np.reshape(gene_expression, [gene_expression.shape[0], 1, gene_expression.shape[1], 1])
    n_input = [1, gene_expression.shape[1]]

    return gene_expression, Img, Label, n_input


class AML(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(170)#需要修改

        data_dir = "./data/"
        data_name1 = "AML_Gene_Expression.txt"
        data_name2 = "AML_Methy_Expression.txt"
        data_name3 = "AML_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img1 = Img1[:, :, 0:558, :]
        Img3 = Img3[:, :, 0:558, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class BIC(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(624)

        data_dir = "./data/"
        data_name1 = "BREAST_Gene_Expression.txt"
        data_name2 = "BREAST_Methy_Expression.txt"
        data_name3 = "BREAST_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:885, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class COAD(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(220)

        data_dir = "./data/"
        data_name1 = "COLON_Gene_Expression.txt"
        data_name2 = "COLON_Methy_Expression.txt"
        data_name3 = "COLON_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:613, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class GBM(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(274)

        data_dir = "./data/"
        data_name1 = "GLIO_Gene_Expression.txt"
        data_name2 = "GLIO_Methy_Expression.txt"
        data_name3 = "GLIO_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:534, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num

class KIRC(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(184)

        data_dir = "./data/"
        data_name1 = "KIDNEY_Gene_Expression.txt"
        data_name2 = "KIDNEY_Methy_Expression.txt"
        data_name3 = "KIDNEY_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:791, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class LIHC(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(367)

        data_dir = "./data/"
        data_name1 = "LIVER_Gene_Expression.txt"
        data_name2 = "LIVER_Methy_Expression.txt"
        data_name3 = "LIVER_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:852, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class LUSC(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(341)

        data_dir = "./data/"
        data_name1 = "LUNG_Gene_Expression.txt"
        data_name2 = "LUNG_Methy_Expression.txt"
        data_name3 = "LUNG_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:881, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class SKCM(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(450)

        data_dir = "./data/"
        data_name1 = "MELANOMA_Gene_Expression.txt"
        data_name2 = "MELANOMA_Methy_Expression.txt"
        data_name3 = "MELANOMA_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:901, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class OV(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(287)

        data_dir = "./data/"
        data_name1 = "OVARIAN_Gene_Expression.txt"
        data_name2 = "OVARIAN_Methy_Expression.txt"
        data_name3 = "OVARIAN_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:616, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class SARC(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(257)

        data_dir = "./data/"
        data_name1 = "SARCOMA_Gene_Expression.txt"
        data_name2 = "SARCOMA_Methy_Expression.txt"
        data_name3 = "SARCOMA_Mirna_Expression.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)
        gene_expression3, Img3, Label3, n_input3 = load_data1(data_name3)


        Img3 = Img3[:, :, 0:838, :]
        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)
        Img3 = np.squeeze(Img3, axis=None)
        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)
        self.Img3 = Img3.astype(np.float32)
        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)
        print(self.Img3.shape)

    def __getitem__(self, index):

        img_traina, img_trainb, img_trainc = self.Img1[index, :], self.Img2[index, :], self.Img3[index, :]
        return [img_traina, img_trainb, img_trainc], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


class METABRIC(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(1904)

        data_dir = "./data/"
        data_name1 = "METABRIC_Gene_Expression.txt"
        data_name2 = "METABRIC_CNV.txt"
        gene_expression1, Img1, Label1, n_input1 = load_data1(data_name1)
        gene_expression2, Img2, Label2, n_input2 = load_data1(data_name2)

        Img1 = np.squeeze(Img1, axis=None)
        Img2 = np.squeeze(Img2, axis=None)

        self.Img1 = Img1.astype(np.float32)
        self.Img2 = Img2.astype(np.float32)

        self.labels = Label1
        print("Img")
        print(self.Img1.shape)
        print(self.Img2.shape)


    def __getitem__(self, index):

        img_traina, img_trainb= self.Img1[index, :], self.Img2[index, :]
        return [img_traina, img_trainb], torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return self.train_num


def load_data(dataset):
    if dataset == "AML":
        dataset = AML('./data/')
        dims = [558, 2000, 558]
        view = 3
        data_size = 170
        class_num = 5
    elif dataset == "BIC":
        dataset = BIC('./data/')
        dims = [2000, 2000, 885]
        view = 3
        data_size = 624
        class_num = 4
    elif dataset == "COAD":
        dataset = COAD('./data/')
        dims = [2000, 2000, 613]
        view = 3
        data_size = 220
        class_num = 5
    elif dataset == "GBM":
        dataset = GBM('./data/')
        dims = [2000, 2000, 534]
        view = 3
        data_size = 274
        class_num = 6
    elif dataset == "KIRC":
        dataset = KIRC('./data/')
        dims = [2000, 2000, 791]
        view = 3
        data_size = 184
        class_num = 2
    elif dataset == "LIHC":
        dataset = LIHC('./data/')
        dims = [2000, 2000, 852]
        view = 3
        data_size = 367
        class_num = 5
    elif dataset == "LUSC":
        dataset = LUSC('./data/')
        dims = [2000, 2000, 881]
        view = 3
        data_size = 341
        class_num = 12
    elif dataset == "SKCM":
        dataset = SKCM('./data/')
        dims = [2000, 2000, 901]
        view = 3
        data_size = 450
        class_num = 5
    elif dataset == "OV":
        dataset = OV('./data/')
        dims = [2000, 2000, 616]
        view = 3
        data_size = 287
        class_num = 4
    elif dataset == "SARC":
        dataset = SARC('./data/')
        dims = [2000, 2000, 838]
        view = 3
        data_size = 257
        class_num = 13
    elif dataset == "METABRIC":
        dataset = METABRIC('./data/')
        dims = [2000, 2000]
        view = 2
        data_size = 1904
        class_num = 9

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
