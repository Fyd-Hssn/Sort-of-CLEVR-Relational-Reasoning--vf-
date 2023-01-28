# Comments for this code are inspired by the Relation Netowrks paper as detailed in "https://arxiv.org/pdf/1706.01427.pdf"

import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Define CNN model class. We need an underlying CNN as the backbone of the RN model to convolve the pixel inputs and feed the resulting objects into the RN. 
class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__() # Allows us to initialize an nn.Module object and utilize its respective methods
        
        # Create four 2-dimensional convolutional layers, each with a stride of 2, padding of 1, and 24 output kernels where each kernel is of size 3x3.
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1) # The first conv layer will filter using 3 kernels/channels on an input image of size 128x128 (pre-process with padding to 136x136) to produce 24 kernels/channels into the next layer  
        self.batchNorm1 = nn.BatchNorm2d(24) # Batch normalization; Normalize the 24 output channels from the preceding layer to reduce the internal covariate shift which will allow for faster training throughout the network.
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # Second conv layer will take 24 input channels and similarly produce 24 kernels/channels onto the next layer
        self.batchNorm2 = nn.BatchNorm2d(24) # Batch normalization; 
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # Third conv layer; same as second
        self.batchNorm3 = nn.BatchNorm2d(24) # Batch normalization; 
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # Fourth/final conv layer; same as second
        self.batchNorm4 = nn.BatchNorm2d(24) # Batch normalization; 

   # Forward pass of the CNN     
    def forward(self, img):
        """convolution"""
        # Inputs to each layer will be of shape (n, h, w, c), where n is the # of data points, c is the # of channels, and h and w are the height/width (intially 128x128).
        x = self.conv1(img) # Convolve the input image starting with the first CNN layer and save output feature maps into x. 
        x = F.relu(x) # Activation function; This will give us linearity (piecewise) and will return max(0, x). This also provides non-diminishing gradients in the backprop as the gradient will always be a constant.
        x = self.batchNorm1(x) # Batch normalization based on output of previous layer. This will regularize the input distribution in the network for the nexy layer.
        
        # The remaining layers simply just upstream the convolution further. Refer to the comments above as the remaining layers/convolutions follow the same code.
        x = self.conv2(x) # Take the normalized output of the first layer and convolve it further.
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x # Return the final set of objects/features that are fed into the RN for relational asssessment.

# This will be the final structure of the RN model, two linear layers that will produce an answer that will produce logits for a softmax over the answer vocabulary. This will be the model for f(φ).
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__() # Allows us to initialize an nn.Module object and utilize its respective methods
        
        # These two layers are part of the 3-layer MLP for f(φ).
        self.fc2 = nn.Linear(256, 256) # First linear (hidden) layer which will take in 256 units from the first fc1 layer (defined in the RN class down below), and feed 256 units into the final layer.
        self.fc3 = nn.Linear(256, 10) # Final layer of the entire RN model, this will output 10 units which will need to be classified (multi-class) for a multinomial probability distribution

    # Forward pass of the FC Model
    def forward(self, x):
        x = self.fc2(x) # Feed the second linear layer with the 256 units produced from the first fc1 layer.
        x = F.relu(x) # Activation function for previous layer. Read previous comments for ReLu (under CNN class) for further insight on its purpose.
        x = F.dropout(x) # 50% dropout (default parameter is p=0.5); Regularization method where in this case 50% of layer outputs are dropped (replaced with zeros) to reduce overfitting and mitigate interrdependency between each neuron.
        x = self.fc3(x) # Pass the remaining layers into the last layer, where output tensor x has 10 units.
        return F.log_softmax(x, dim=1) # Compute softmax on the x tensor on 1 dimension (log calculated after softmax). Returns a final tensor with same dimensionality as input tensor to the softmax with value range of [-inf, 0].

# BasicModel class will define the training/testing functionality which the RN model will operate on.
class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__() # Allows us to initialize an nn.Module object and utilize its respective methods
        self.name=name # Simply assigns the name of the model. In our case the name will always be 'RN' as we will not use the CNN-MLP model as an comparable in this project. 

    # Training process functionality
    def train_(self, input_img, input_qst, label):
        '''
        self = model object used to call train_()
        input_img = instance of SORT-of-CLEVR image we will train on
        input_qst = question that the RN will have to answer
        label = Tensor of minibatch size (64); Refer to main.py for more context
        '''
        self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero
        output = self(input_img, input_qst) # Model will release a tensor of (n, c) dimensions where n is the # of data points and c is the number of classes
        loss = F.nll_loss(output, label) # Compute loss on output tensor, using label as the target sensor which is (64) in our case. 
        loss.backward() # Back propagation of the RN. Computes gradient of the loss function.
        self.optimizer.step() # Perform a single optimization step / parameter update.
        pred = output.data.max(1)[1] # Set the prediction equal to the max of all the elements in the output tensor.
        correct = pred.eq(label.data).cpu().sum() # Compute the element-wise equality of the prediction vs. the label tensor, which returns a boolean tensor where each element will be T/F based on comparison. Next, we compute the sum of all boolean values. cpu() just copys the tensor to CPU memory.
        accuracy = correct * 100. / len(label) # Compute accuracy against the label.
        return accuracy, loss 
        
    # Testing process functionality. Refer to comments above for train_() as code is the same. 
    def test_(self, input_img, input_qst, label):
        '''
        self = model object used to call test_()
        input_img = instance of SORT-of-CLEVR image we will train on
        input_qst = question that the RN will have to answer
        label = Tensor of minibatch size (64); Refer to main.py for more context
        '''

        # The difference in purpose from train_() is that we will call test_() on untrained SORT-of-CLEVR images to evaulate our model's accuracy and performance. 
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    # Save the current model's state from the current epoch into a PTH file. This will store a snapshot of the model's tensor information in a dictionary as well as its parameters. 
    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))

# RN Model Class which is a child class of BasicModel() above. This will provide the structure to run relational reasoning via the compsite function: RN(O) = fφ(Σ[gθ(oi, oj)]).
class RN(BasicModel):
    def __init__(self, args):
        '''
        args = a set of parameters that we define in the terminal for the model to parse through
        '''
        super(RN, self).__init__(args, 'RN') # Allows us to initialize an BasicModel object and utilize its training/testing methods.
        
        self.conv = ConvInputModel() # Instantiate the CNN portion of the model to convolve the image dataset
        
        self.relation_type = args.relation_type # Parse the relation type that we are trying to infer on (binary or ternary).
        
        # Set up the four-layer MLP for gθ, where each layer will produce 256 units with ReLU non-linearities 
        if self.relation_type == 'ternary':
            ##(number of filters per object+coordinate of object)*3+question vector
            self.g_fc1 = nn.Linear((24+2)*3+18, 256) # For ternary, we will multiply the base input units by a factor of 3
        else:
            ##(number of filters per object+coordinate of object)*2+question vector
            self.g_fc1 = nn.Linear((24+2)*2+18, 256) # For ternary, we will multiply the base input units by a factor of 2

        # Set up the remaining MLP layers for gθ. These layers will be used to evaluate relational scores for each object pair & question set
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        # Set the first FC layer for fφ. We will utilize the FC model to derive our answer on the question set based on the relational scores from the gθ MLP layers
        self.f_fc1 = nn.Linear(256, 256)

        # Set the tensors for each object in the relational pair
        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        
        # Check to see if we are performing the RN on the GPU. If so, set tensor to reference object placed in GPU memory
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()

        # Use the Variable wrapper around tensors to easily allow gradient computation through autograd    
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # Prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        # Coordinate tensor matrix of shape (args.batch_size, 25, 2)
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        
        # Create numpy matrix resembling the shape of coord_tensor which we will use to calculate the necessary coordinates 
        np_coord_tensor = jnp.zeros((args.batch_size, 25, 2))
        
        # Loop through and rewrite the numpy matrix to have the new coordinate values
        for i in range(25):
            np_coord_tensor[:,i,:] = jnp.array( cvt_coord(i) )

        # Rewrite coord_tensor and copy the calculated coordinate values
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        # Remaining MLP layers for fφ. This will output the final answer for a given question and object pair. See comments above for FCOutputModel() for further insight. 
        self.fcout = FCOutputModel()
        
        # Softmax output from FCOutputModel() is optimized with a cross-entropy loss function using the Adam optimizer below. Learning Rate per the paper is 2.5e^-4. 
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    # Forward pass of the RN model. Dimensions of tensors are already provided by the author below. 
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        print(x.size())
        '''Forward pass for gθ''' 
        mb = x.size()[0] # Minibatch Size; Paper used batch size of 64 throughout
        n_channels = x.size()[1] # Number of input features 
        d = x.size()[2] # d*d is equal to feature map dimensions (5x5)
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1) # Composes an x_flat tensor by altering the dimensionality of x and permutates the dimensions indices to be in the form (mb, d*d, n_channels). Total # of elements remains the same in both tensors.
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2) # Concatenates the x_flat and coordinate tensors in the last dimension
        
        # Feature concatenation
        if self.relation_type == 'ternary':
            # Add question everywhere; The RN is conditioned with question embeddings such that a = fφ(Σi,j [gθ(oi, oj , q)]), where q is the encoded question
            qst = torch.unsqueeze(qst, 1) # (64x1x18); # Takes the qst tensor and returns a new tensor with dim(1) inserted at position 1 of the original qst.shape()
            qst = qst.repeat(1, 25, 1) # (64x25x18) # Repeats tensor along specified dimensions.
            qst = torch.unsqueeze(qst, 1)  # (64x1x25x18); ''
            qst = torch.unsqueeze(qst, 1)  # (64x1x1x25x18); ''

            # Cast all triples against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_i = torch.unsqueeze(x_i, 3)  # (64x1x25x1x26)
            x_i = x_i.repeat(1, 25, 1, 25, 1)  # (64x25x25x25x26)
            
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26)
            x_j = torch.unsqueeze(x_j, 2)  # (64x25x1x1x26)
            x_j = x_j.repeat(1, 1, 25, 25, 1)  # (64x25x25x25x26)

            x_k = torch.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_k = torch.unsqueeze(x_k, 1)  # (64x1x1x25x26)
            x_k = torch.cat([x_k, qst], 4)  # (64x1x1x25x26+18)
            x_k = x_k.repeat(1, 25, 25, 1, 1)  # (64x25x25x25x26+18)

            # Concatenate all together
            x_full = torch.cat([x_i, x_j, x_k], 4)  # (64x25x25x25x3*26+18)

            # Reshape for passing through network; x_ will the set of all object pairs for the gθ-MLP layers to process
            x_ = x_full.view(mb * (d * d) * (d * d) * (d * d), 96)  # (64*25*25*25, 3*26+18) = (1.000.000, 96); d*d is the size of the feature maps for each object
        
        # Concatenation step for binary relations
        else:
            # Add question everywhere; The RN is conditioned with question embeddings such that a = fφ(Σi,j [gθ(oi, oj , q)]), where q is the encoded question
            qst = torch.unsqueeze(qst, 1) # (64x1x18)
            qst = qst.repeat(1, 25, 1) # (64x25x18)
            qst = torch.unsqueeze(qst, 2)

            # Cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+18)
            x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x26+18)
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+18)
            x_j = torch.cat([x_j, qst], 3)
            x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x26+18)
            
            # Concatenate all together
            x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+18)
        
            # Reshape for passing through network; x_ will the set of all object pairs for the gθ-MLP layers to process
            x_ = x_full.view(mb * (d * d) * (d * d), 70)  # (64*25*25, 2*26*18) = (40.000, 70); d*d is the size of the feature maps for each object
        
        '''gθ-MLP structure; Assess relational score between objects (ternary or binary)'''
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_) # x_ is the final set relational scores for each object pair passed into gθ. We now need to compute the element-wise sum to pass into fφ-MLP.
        
        # Reshape again and sum
        if self.relation_type == 'ternary':
            x_g = x_.view(mb, (d * d) * (d * d) * (d * d), 256)
        else:
            x_g = x_.view(mb, (d * d) * (d * d), 256)

        # Element-wise sum of relational scores for each object pair
        x_g = x_g.sum(1).squeeze()
        
        """fφ-MLP"""
        x_f = self.f_fc1(x_g) # Pass the element-wise sum tensor into the first layer of the fφ-MLP structure.
        x_f = F.relu(x_f) # Activation function
        
        '''Pass into final two layers of FC network to calculate and return the softmax of all the relational scores, giving us the answer of a question on a given set of objects.'''
        return self.fcout(x_f) 

# Ignore CNN_MLP class as we will not be running it for our use case. However, code is kept to avoid dependency issues.
class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 18, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)