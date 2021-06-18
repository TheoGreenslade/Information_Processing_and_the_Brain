import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import expit

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

def readInDatasets():
    print("--loading data--")
    with open("pickled_mnist.pkl", "br") as fh:
        data = pickle.load(fh)
    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]
    train_labels_one_hot = data[4]
    test_labels_one_hot = data[5]
    print("--data loaded--")
    return train_imgs, test_imgs, train_labels, test_labels, train_labels_one_hot, test_labels_one_hot

def generate_batch(imgs, labels, batch_size):
    #differentiate inputs (features) from targets and transform each into 
    #numpy array with each row as an example
    # images = np.vstack([ex for ex in imgs])
    # labels = np.vstack([ex for ex in labels])
    images = imgs
    
    #randomly choose batch_size many examples; note there will be
    #duplicate entries when batch_size > len(dataset) 
    rand_inds = np.random.randint(0, len(labels), batch_size)
    images_batch = images[rand_inds]
    labels_batch = labels[rand_inds]
    
    return images_batch, labels_batch

def sigmoid(a):
    # siga = 1/(1 + np.exp(-a))
    siga = expit(a)
    return siga

class nn_one_layer():
    def __init__(self, input_size, hidden_size, output_size):
        #define the input/output weights W1, W2
        self.W1 = 0.1 * np.random.randn(input_size, hidden_size)
        self.W2 = 0.1 * np.random.randn(hidden_size, output_size)
        
        self.f = sigmoid
    
    def forward(self, u):
        # z = np.matmul(u, self.W1, dtype=np.float128)
        # h = self.f(z)
        # v = np.matmul(h, self.W2)
        z = np.matmul(u, self.W1, dtype=np.float128)
        h1 = self.f(z)
        h2 = np.matmul(h1, self.W2)
        indices = np.argmax(h2, axis=1)
        v = np.ones(h2.shape) * 0.01
        for i in range(len(indices)):
            v[i][indices[i]] = 0.99
        # v =  h2
        return v, h1, z

def plotPedictions(images, labels,predictions, inds):
    fig = plt.figure()
    # ax1 = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=2)
    # plt.scatter(labels[inds], predictions[inds], marker='x', c='black')
    # plt.xlabel('target values')
    # plt.ylabel('predicted values')
    # plt.ylim([np.min(preds_xor) - 0.01, np.max(preds_xor) + 0.01])
    # ax2 = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=1)
    # plt.hist(targets_xor, color='blue')
    # ax2.set_title('target values')
    # plt.ylabel('# in batch')
    # ax3 = plt.subplot2grid((2,2), (1,1), rowspan=1, colspan=1, sharey=ax2)
    # plt.hist(preds_xor, color='red')
    # ax3.set_title('predicted values')

    # fig.tight_layout()

#loss function as defined above
def loss_mse(preds, targets):
    loss = np.sum((preds - targets)**2)
    return 0.5 * loss

# #derivative of loss function with respect to predictions
def loss_deriv(preds, targets):
    dL_dPred = preds - targets
    return dL_dPred

# #derivative of the sigmoid function
def sigmoid_prime(a):
    dsigmoid_da = sigmoid(a)*(1-sigmoid(a))
    return dsigmoid_da

#compute the derivative of the loss wrt network weights W1 and W2
#dL_dPred is (precomputed) derivative of loss wrt network prediction
#X is (batch) input to network, H is (batch) activity at hidden layer
def backprop(W1, W2, dL_dPred, U, H, Z):
    #hints: for dL_dW1 compute dL_dH, dL_dZ first.
    #for transpose of numpy array A use A.T
    #for element-wise multiplication use A*B or np.multiply(A,B)
    
    dL_dW2 = np.matmul(H.T, dL_dPred)
    dL_dH = np.matmul(dL_dPred, W2.T)
    dL_dZ = np.multiply(sigmoid_prime(Z), dL_dH)
    dL_dW1 = np.matmul(U.T, dL_dZ)
    
    return dL_dW1, dL_dW2

def calculateAccuracy(preds, labels):
    sum = 0
    for i in range(len(labels)):
        same = True
        for j in range(10):
            if preds[i][j] != labels[i][j]:
                same = False
        if same:
            sum = sum + 1 
    accuracy = (sum / labels.shape[0])* 100
    return accuracy

def testAccuracy(nn, imgs, labels):
    inputs, targets = generate_batch(imgs, labels, batch_size=200)
    preds, _, _ = nn.forward(inputs) 
    accuracy = calculateAccuracy(preds, targets)
    return accuracy


#train the provided network with one batch according to the dataset
#return the loss for the batch
def train_one_batch(nn, imgs, labels, batch_size, lr):
    inputs, targets = generate_batch(imgs, labels, batch_size)
    preds, H, Z = nn.forward(inputs)
 
    loss = loss_mse(preds, targets)

    dL_dPred = loss_deriv(preds, targets)
    dL_dW1, dL_dW2 = backprop(nn.W1, nn.W2, dL_dPred, U=inputs, H=H, Z=Z)

    nn.W1 -= lr * dL_dW1
    nn.W2 -= lr * dL_dW2
    
    return loss

def test(nn, imgs, labels):
    inputs, targets = generate_batch(imgs,labels, batch_size=200)
    preds, _, _ = nn.forward(inputs) 
    loss = loss_mse(preds, targets)
    return loss

# ------------------------------------------------------------------------------------------------------------------

train_imgs, test_imgs, train_labels, test_labels, train_labels_one_hot, test_labels_one_hot = readInDatasets()

# batch_images, batch_labels = generate_batch(train_imgs,train_labels, batch_size=100)
# print(train_images)
# print(train_labels)

input_size = image_pixels
hidden_size = 100
output_size = 10

nn = nn_one_layer(input_size, hidden_size, output_size) #initialise model
# predictions, _, _ = nn.forward(batch_images) #prediction made by model on batch xor input
# _, inds = np.unique(batch_images, return_index=True, axis=0)

# plotPedictions(train_images,train_labels, predictions, inds) ----- finish if needed

batch_size = 5 #number of examples per batch
nbatches = 5000 #number of batches used for training
lr = 0.1 #learning rate

losses = [] #training losses to record
accuracies = []
print('--calculating losses--')
for i in range(nbatches):
    step = str(i) + ' '
    # print(step, end='', flush=True)
    loss = train_one_batch(nn, train_imgs, train_labels_one_hot, batch_size=batch_size, lr=lr)
    # accuracy = testAccuracy(nn,test_imgs,test_labels_one_hot)
    losses.append(loss)
    # accuracies.append(accuracy)
print('--losses calculated--')

# plt.plot(np.arange(1, nbatches+1), losses)
# plt.xlabel("# batches")
# plt.ylabel("training MSE")
# plt.show()

# plt.plot(np.arange(1, nbatches+1), accuracies)
# plt.xlabel("# batches")
# plt.ylabel("training MSE")
# plt.show()

# inputs, targets = generate_batch(test_imgs, test_labels_one_hot, batch_size=200)
# preds, _, _ = nn.forward(inputs) 
# accuracy = calculateAccuracy(preds,targets)
# print(accuracy)

accuracy = testAccuracy(nn, test_imgs, test_labels_one_hot)
print(accuracy)

# inputs, targets = generate_batch(train_imgs, train_labels, batch_size=100)
# preds, _, _ = nn.forward(inputs) #prediction made by model

# _, inds = np.unique(inputs, return_index=True, axis=0)





# fig = plt.figure()
# ax1 = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=2)
# plt.scatter(targets[inds], preds[inds], marker='x', c='black')

# fig = plt.figure()
# ax1 = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=2)
# plt.scatter(targets[inds], preds[inds], marker='x', c='black')

# yup = 0.1
# ydown = -0.1
# for i in inds:
#     coord = '({}, {})'.format(inputs[i][0], inputs[i][1])
#     if np.isclose(preds[i], 0, atol=0.1):
#         yup = 2 * yup
#         yoffset = yup
#     else:
#         ydown = 2 * ydown
#         yoffset = ydown
    
#     xoffset = 0.05 if targets[i] == 0 else -0.1
#     plt.text(targets[i] + xoffset, preds[i] + yoffset, coord)
# plt.xlabel('target values')
# plt.ylabel('predicted values')
# plt.ylim([np.min(preds) - 0.1, np.max(preds) + 0.1])
# ax2 = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=1)
# plt.hist(targets, color='blue')
# ax2.set_title('target values')
# plt.ylabel('# in batch')
# ax3 = plt.subplot2grid((2,2), (1,1), rowspan=1, colspan=1, sharey=ax2)
# plt.hist(preds, color='red')
# ax3.set_title('predicted values')

# dataset_names = ['AND gate', 'OR gate', 'XOR gate', 'XNOR gate']
# test_scores = [test(nn, dataset) for dataset in [dataset_and, 
#                             dataset_or, dataset_xor, dataset_xnor]]

# x = range(4)
# plt.bar(x, test_scores)
# plt.xticks(x, dataset_names, rotation='vertical')
# plt.ylabel("test MSE")


