import numpy as np
from PIL import Image

class Proxy_Data():
    def __init__(self, test_transform=None):
        super(Proxy_Data, self).__init__()
        self.test_transform = test_transform
        self.TestData = np.array([])
        self.TestLabels = np.array([])

    def concatenate(self, datas, labels):
        if not datas:
            return np.array([]), np.array([])
            
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, new_set, new_set_label):
        datas, labels = [], []
        self.TestData = np.array([])
        self.TestLabels = np.array([])
        
        if len(new_set) != 0 and len(new_set_label) != 0:
            datas = [exemplar for exemplar in new_set]
            for i in range(len(new_set)):
                length = len(datas[i])
                labels.append(np.full((length,), new_set_label[i]))

        if datas:
            self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        return img, target

    def __getitem__(self, index):
        if self.TestData.size > 0:
            return self.getTestItem(index)

    def __len__(self):
        test_data_array = np.array(self.TestData)
        if test_data_array.size > 0:
            return len(self.TestData)
        return 0