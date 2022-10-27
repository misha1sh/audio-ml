class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential( # 1, 128, 9
            nn.Conv2d(1, 4, kernel_size=3, padding="same"),  # 4, 128, 9
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(4, 1)),  # 4, 32, 9

            nn.Conv2d(4, 4, kernel_size=(4, 3), padding="same"), # 4, 32, 9
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(4, 1)),  # 4, 8, 9

            # nn.Tanhshrink(),
            # nn.Sigmoid(),
            # nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(64, 128),
            # nn.ReLU(),

            # nn.ConvTranspose2d(4, 4, kernel_size=(8, 1), stride=(4, 1)), # 4, 32, 9
            # nn.ConvTranspose2d(4, 1, kernel_size=(4, 1), stride=(4, 1)),  # 1, 128, 9

            nn.UpsamplingNearest2d(scale_factor=(4, 1)),  # 4, 8, 3
            nn.Conv2d(4, 4, kernel_size=(4, 1), padding="same"), # 4, 32, 9
            nn.Tanh(),
            nn.UpsamplingNearest2d(scale_factor=(4, 1)), # 4, 128, 9
            nn.Conv2d(4, 1, kernel_size=(4, 4), padding="same"), # 1, 128, 9
            nn.Tanh(),
            # nn.UpsamplingNearest2d(scale_factor=(4, 1)),
            # nn.Conv2d(4, 1, kernel_size=5, padding="same"),
            # nn.Tanhshrink(),

            # nn.Unflatten(1, (1, 128, 9))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AE()
cuda0 = torch.device('cuda:0')

x_gpu = x.to(cuda0)
# add dimension
x_gpu = torch.unsqueeze(x_gpu, 1)
model_gpu = model.to(cuda0)

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-2 * 5)
                            #  weight_decay = 1e-8)
scheduler = ReduceLROnPlateau(optimizer, factor=0.9, threshold=1e-4, patience=10)

epochs = 10000
outputs = []
losses = []
for epoch in range(epochs):

    #   # Reshaping the image to (-1, 784)
    #   image = image.reshape(-1, 28*28)

    # Output of Autoencoder
    reconstructed = model_gpu(x_gpu)
    # print(reconstructed.shape)

    # Calculating the loss function
    loss = loss_function(reconstructed, x_gpu)

    # The gradients are set to zero,
    # the gradient is computed and stored.
    # .step() performs parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Storing the losses in a list for plotting
    l = loss.cpu().detach().numpy()
    losses.append(l)

    scheduler.step(l)
    lr = optimizer.param_groups[0]['lr']
    print(l, lr)
    del reconstructed
    del loss

    if lr < 1e-5 or l < 0.01:
        print("break at", epoch)
        break
    # audio = spectrogrammer.mel2wave(model(x[0:1]))
    # Audio(audio.numpy(), sample_rate=SAMPLE_RATE)

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(losses[100:])