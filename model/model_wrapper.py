'''
Wrapper for the model to add the final layer for classification

code is adapted from github.com/conditionWang/FCIL

'''
import torch.nn as nn

class ClassificationWrapper(nn.Module):
    '''
    Wrapper for the model to add the final layer for classification
    '''
    def __init__(self, model, num_classes):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
    
    def incremental_learning(self, num_classes):
        '''
        Summary: Incrementally learn new classes

        Args:
            num_classes (int): Number of new classes to learn

        How does it work?
        - save the weight and bias of the fully connected layer
        - create a new fully connected layer with the new number of classes
        - copy the saved weight and bias to the new fully connected layer

        Why is it important?
        - because we want to incrementally learn new classes without forgetting the old classes
        - so we don't need to retrain the whole model, only the last layer
        '''
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        # create a new fully connected layer
        self.fc = nn.Linear(in_feature, num_classes, bias=True)
        # copy the saved weight and bias to the new fully connected layer
        self.fc.weight.data[:out_feature] = weight
        # copy the saved bias to the new fully connected layer
        self.fc.bias.data[:out_feature] = bias

    def features_extractor(self, x):
        '''
        Summary: Extract features from the model
        '''
        return self.model(x)
    
    def predict(self, x):
        '''
        Summary: Predict the class of the input
        '''
        return self.fc(x)