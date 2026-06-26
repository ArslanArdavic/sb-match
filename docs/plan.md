#  Download the dataset 

-[Animal Faces-HQ (AFHQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
- 16,130 high-quality images at 512×512 resolution
- Three domains of classes, each providing about 5000 images
    - Cat
    - Dog
    - Wildlife

# Split the dataset

- Pick `n` images of class Cat 
    - Keep `m` for test
    - Keep `k` for evaluation
    - Keep `n - (m + k)`
- Pick `n` images of class Wildlife 
    - Keep `m` for test
    - Keep `k` for evaluation
    - Keep `n - (m + k)`

- Determine the values `n`, `m`, `k` selected in the baseline method

# Define the methods

- Store data coupling, initially uniform random couple

- Define the reference stochastic process 
- Simulate the reference stochastic process starting from a data instance (sanity check)

- Condition the reference process with endpoints 
- Simulate the endpoint-conditioned bridge (sanity check)
- Sample an intermediate state from endpoint-conditioned bridge 

- Sample the markovian drift regression target
- Predict a markovian drift using a neural network
- Simulate the markovian projection SDE with the latest drift and store the new coupling

# Setup 1. Parameterize only the forward-time Markov control drift

- Define a single neural net that parameterizes the forward-time Markov control drift
- Write the training loop and execute
- Pick test samples of source dataset and simulate forward markovian projection SDE 

# Setup 2. Parameterize both the forward-time and reverse-time Markovian drifts 

- Define two neurals net parameterizing
    - the forward-time Markov control drift 
    - the reverse-time Markov control drift
- Write the alternating training loop and execute
- Pick source dataset test samples  and simulate forward markovian projection SDE 
- Pick target dataset test samplest and simulate reverse markovian projection SDE 

# Setup 3. Joint Training of Forward and Reverse Markovian Projections

- Define two neurals net parameterizing
    - the forward-time Markov control drift 
    - the reverse-time Markov control drift
- Write the alternating training loop including the consistency loss and execute
- Pick source dataset test samples  and simulate forward markovian projection SDE 
- Pick target dataset test samplest and simulate reverse markovian projection SDE 