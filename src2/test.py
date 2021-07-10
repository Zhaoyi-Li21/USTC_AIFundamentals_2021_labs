from MLP_manual import MLP
import numpy as np
input_nodes = 5
hidden1_nodes = 4
hidden2_nodes = 4
output_nodes = 3
epochs = 100
learning_rate = 0.1

mlp = MLP(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate,epochs)

inputs = np.matrix([1.0, 0.5, -1.5, 1.5, -0.5])
# inputs = np.zeros()
print(mlp.query(inputs))


