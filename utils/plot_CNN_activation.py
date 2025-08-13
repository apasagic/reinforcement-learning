from tensorflow.keras import Input
from tensorflow.keras import layers, Model, Sequential

n = 0

inputs = Input(shape=(84, 84, 4))
x1 = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
x2 = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x1)
x3 = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x2)

debug_model = Model(inputs=inputs, outputs=[x1, x2, x3])

# 2. Copy weights from original model
for i in range(3):  # First 3 Conv2D layers
    debug_model.layers[i+1].set_weights(self.QNetwork.layers[i].get_weights())

state = states[n,:,:,:].reshape((1,84,84,4))
layer_outputs = debug_model.predict(state, verbose=0)

# Activation layer_outputs
out1 = layer_outputs[0].reshape((1, 20, 20, 32))
out2 = layer_outputs[1].reshape((1, 9, 9, 64))
out3 = layer_outputs[2].reshape((1, 7, 7, 64))

plt.figure(figsize=(12, 4))
plt.imshow(state[n,:,:,0], cmap='gray')
plt.title('Input State')

#plt.figure(figsize=(12, 4))
#plt.subplot(1, 3, 1)
#plt.imshow(out3[0,:,:,0], cmap='gray')

#Plot the activations of the first convolutional layer
# Assuming `out1` is your first conv layer output (shape: (1, 20, 20, 32))
activation = out3[0]  # remove batch dimension -> shape becomes (20, 20, 32)

num_filters = activation.shape[-1]  # 32
cols = 8                            # number of columns in plot grid
rows = int(np.ceil(num_filters / cols))

plt.figure(figsize=(cols * 2, rows * 2))

for i in range(num_filters):
    ax = plt.subplot(rows, cols, i + 1)
    plt.imshow(activation[:, :, i], cmap='viridis')
    plt.axis('off')
    ax.set_title(f'Filter {i}', fontsize=8)

plt.tight_layout()
plt.show()