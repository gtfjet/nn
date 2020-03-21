import math, random
random.seed(67676776)
# A node in a neural network. Each node has a state
# (total input, output, and their respectively derivatives) which changes
# after every forward and back propagation run.
class Node:
   # Creates a new node with the provided id.
  def __init__(self,id):
    # List of input links. 
    self.inputLinks = []
    self.bias = 0.1
    # List of output links. 
    self.outputs = []
    self.totalInput = 0
    self.output = 0
    # Error derivative with respect to this node's output. 
    self.outputDer = 0
    # Error derivative with respect to this node's total input. 
    self.inputDer = 0
     # Accumulated error derivative with respect to this node's total input since
     # the last update. This derivative equals dE/db where b is the node's
     # bias term.
    self.accInputDer = 0
     # Number of accumulated err. derivatives with respect to the total input
     # since the last update.
    self.numAccumulatedDers = 0
    self.id = id
  # Recomputes the node's output and returns it. 
  def updateOutput(self):
    # Stores total input into the node.
    self.totalInput = self.bias
    for link in self.inputLinks:
      self.totalInput += link.weight * link.source.output
    self.output = math.tanh(self.totalInput)
    return self.output

# A link in a neural network. Each link has a weight and a source and
# destination node. Also it has an internal state (error derivative
# with respect to a particular input) which gets updated after
# a run of back propagation.
class Link:
   # Constructs a link in the neural network initialized with random weight.
   # @param source The source node.
   # @param dest The destination node.
  def __init__(self,source,dest):
    self.weight = random.uniform(0,1) - 0.5
    # Error derivative with respect to this weight. 
    self.errorDer = 0
    # Accumulated error derivative since the last update. 
    self.accErrorDer = 0
    # Number of accumulated derivatives since the last update. 
    self.numAccumulatedDers = 0
    self.id = source.id + "-" + dest.id
    self.source = source
    self.dest = dest

# Builds a neural network.
# @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
#   the network will have one input node, 2 nodes in first hidden layer,
#   3 nodes in second hidden layer and 1 output node.
def buildNetwork(networkShape):
  numLayers = len(networkShape)
  id = 1
  # List of layers, with each layer being a list of nodes. 
  network = []
  for layerIdx in range(numLayers):
    isOutputLayer = layerIdx == numLayers - 1
    isInputLayer = layerIdx == 0
    currentLayer = []
    network.append(currentLayer)
    numNodes = networkShape[layerIdx]
    for i in range(numNodes):
      nodeId = str(id)
      id+=1
      node = Node(nodeId)
      currentLayer.append(node)
      if (layerIdx >= 1):
        # Add links from nodes in the previous layer to this node.
        for prevNode in network[layerIdx-1]:
          link = Link(prevNode, node)
          prevNode.outputs.append(link)
          node.inputLinks.append(link)
  return network

# Runs a forward propagation of the provided input through the provided
# network. This method modifies the internal state of the network - the
# total input and output of each node in the network.
# @param network The neural network.
# @param inputs The input array. Its length should match the number of input
#     nodes in the network.
# @return The final output of the network.
def forwardProp(network, inputs): 
  inputLayer = network[0]
  if (len(inputs) != len(inputLayer)):
    print("The number of inputs must match the number of nodes in" +
        " the input layer")
  # Update the input layer.
  for i in range(len(inputLayer)):
    node = inputLayer[i]
    node.output = inputs[i]
  for layerIdx in range(1,len(network)):
    currentLayer = network[layerIdx]
    # Update all the nodes in this layer.
    for node in currentLayer:
      node.updateOutput()
  return network[len(network)-1][0].output

# Runs a backward propagation using the provided target and the
# computed output of the previous call to forward propagation.
# This method modifies the internal state of the network - the error
# derivatives with respect to each node, and each weight
# in the network.
def backProp(network, target):
  # The output node is a special case. We use the user-defined error
  # function for the derivative.
  outputNode = network[len(network)-1][0]
  outputNode.outputDer = outputNode.output-target
  # Go through the layers backwards.
  for layerIdx in reversed(range(1,len(network))):
    currentLayer = network[layerIdx]
    # Compute the error derivative of each node with respect to:
    # 1) its total input
    # 2) each of its input weights.
    for node in currentLayer:
      node.inputDer = node.outputDer * (1 - node.output**2)
      node.accInputDer += node.inputDer
      node.numAccumulatedDers+=1
    # Error derivative with respect to each weight coming into the node.
    for node in currentLayer:
      for link in node.inputLinks:
        link.errorDer = node.inputDer * link.source.output
        link.accErrorDer += link.errorDer
        link.numAccumulatedDers+=1
    if (layerIdx == 1):
      continue
    prevLayer = network[layerIdx-1]
    for node in prevLayer:
      # Compute the error derivative with respect to each node's output.
      node.outputDer = 0
      for output in node.outputs:
        node.outputDer += output.weight * output.dest.inputDer

# Updates the weights of the network using the previously accumulated error
# derivatives.
def updateWeights(network, learningRate):
  for layerIdx in range(1,len(network)):
    currentLayer = network[layerIdx]
    for node in currentLayer:
      # Update the node's bias.
      if (node.numAccumulatedDers > 0):
        node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers
        node.accInputDer = 0
        node.numAccumulatedDers = 0
      # Update the weights coming into this node.
      for link in node.inputLinks:
        if (link.numAccumulatedDers > 0):
          # Update the weight based on dE/dw.
          link.weight -= learningRate * link.accErrorDer / link.numAccumulatedDers
          link.accErrorDer = 0
          link.numAccumulatedDers = 0

def printSolution(network):
  eqn = '\n\n'
  for layerIdx in range(1,len(network)):
    currentLayer = network[layerIdx]
    for node in currentLayer:
      eqn += 'x' + str(node.id) + '=tanh('
      for link in node.inputLinks:
        eqn += str(link.weight) + '*x' + link.source.id + ' + '
      eqn += str(node.bias) + ');\n'
  print(eqn)

def spiralData(numSamples):
  points=[]
  n = numSamples / 2
  def genSpiral(deltaT, label):
    for i in range(n):
      r = float(i) / n * 5
      t = 1.75 * i / n * 2 * math.pi + deltaT
      x = r * math.sin(t)
      y = r * math.cos(t)
      points.append([x, y, label])
  genSpiral(0, 1) #Positive examples.
  genSpiral(math.pi, -1) #Negative examples.
  return points

#main
net=buildNetwork([2,6,1])
points=spiralData(500)
random.shuffle(points)
m=0
for n in range(3000):
  loss=0
  for point in points:
    m+=1
    [x,y,z]=point
    zp=forwardProp(net,[x,y])
    backProp(net,z)
    loss += 0.5 * (zp - z)**2
    if m%10 == 0:
      updateWeights(net,0.1)
  loss = loss/len(points)
  print(n,loss)
printSolution(net)