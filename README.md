# Parametric identification of adaptive activation functions in pre-trained neural network models

Implementation of the experiment as published in the paper.

## Goals of the experiment

The experiment:

- demonstrates the method of activation function replacement in pre-trained
  models using the VGG-like KerasNet [^1] CNN as the example base model when the
  replacement activation function is NOT a generalization of the activation
  function in the base model;
- evaluates the inference result differences between the base pre-trained model
  and the same model with replaced activation functions;
- demonstrates the effectiveness of activation function fine-tuning when all
  other elements of the model are fixed (frozen);
- evaluates performance of the KerasNet variants with different activation
  functions (adaptive and non-adaptive) trained in different regimes;
- demonstrates how adaptive activation functions (LEAF [^5], AHAF [^6], the
  F-Neuron Activation [^7]) adapt their form during fine-tuning to better match
  the rest of the model parameters.

## Description of the experiment

The experiment consists of the following steps:

1. Train the base KerasNet network on the CIFAR-10 [^2] dataset for 100 epochs
   using the standard training procedure and ADAM. The baseline includes 2
   model variants: with ReLU [^3] and SiLU [^4] activation functions across all
   network layers. The baseline network is then saved for further experiments,
   its performance is evaluated on the test subset of CIFAR-10.
2. Load the baseline network with ReLU activations. Create two variants of the
   network. For variant 1 - replace all ReLU activations with AHAF-as-ReLU, for
   variant 2 - with LEAF-as-ReLU. Complete the fine-tuning process with all
   non-AAF weights fixed. Evaluate the network performance on CIFAR-10.
3. Load the baseline network with SiLU activations. Create two variants of the
   network. For variant 1 - replace all SiLU activations with AHAF-as-SiLU, for
   variant 2 - with LEAF-as-SiLU. Complete the fine-tuning process with all
   non-AAF weights fixed. Evaluate the network performance on CIFAR-10.
4. Load the baseline network with ReLU activations. Create three variants of the
   network. Replace the ReLU activations in the fully connected layers with the
   F-Neuron activation initialized as Tanh (variant 1), LEAF-as-Tanh (variant 2),
   and the F-Neuron activation initialized randomly (variant 3).
   For variants 1 and 3 leave the original ReLU activations in the CNN layers. For
   variant 2 replace the activation functions in CNN layers with LEAF-as-ReLU.
   Complete the fine-tuning process with all non-AAF weights fixed. Evaluate the
   network performance on CIFAR-10. 
5. Load the baseline network with SiLU activations. Create three variants of the
   network. Replace the ReLU activations in the fully connected layers with the
   F-Neuron activation initialized as Tanh (variant 1), LEAF-as-Tanh (variant 2),
   and the F-Neuron activation initialized randomly (variant 3).
   For variants 1 and 3 leave the original SiLU activations in the CNN layers. For
   variant 2 replace the activation functions in CNN layers with LEAF-as-SiLU.
   Complete the fine-tuning process with all non-AAF weights fixed. Evaluate the
   network performance on CIFAR-10.
6. Load the baseline network with SiLU activations. Create two variants of the
   network. Replace the SiLU activations across all layers with the
   AHAF (variant 1) and LEAF (variant 2) initialized as ReLU. Complete the
   fine-tuning process with all non-AAF weights fixed. Evaluate the network
   performance on CIFAR-10. Evaluate the network performance on CIFAR-10. 
7. Load the baseline network with ReLU activations. Create two variants of the
   network. Replace the ReLU activations across all layers with the AHAF
   (variant 1) and LEAF (variant 2) initialized as SiLU. Complete the
   fine-tuning process with all non-AAF weights fixed. Evaluate the network
   performance on CIFAR-10. Evaluate the network performance on CIFAR-10. 

## Running experiments

1. NVIDIA GPU recommended with at least 2 GiB of VRAM.
2. Install the requirements from `requirements.txt`.
3. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment variables.
4. Use the root of this repository as the current directory.
5. Add the current directory to `PYTHONPATH` so it can find the modules

This repository contains a wrapper script that sets all the required
environment variables: [run_experiment.sh](./run_experiment.sh). Use the bash shell to
execute the experiment using the wrapper script:

Example:

```shell
user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py  #...
```

## Reproducing results from the paper

Execute the [run_experiment_all.sh](./run_experiment_all.sh) script to perform
all experiments in an automated way. For parallel execution - consult the
list of commands from this script and execute them manually.

Example:

```shell
user@host:~/repo_path$ ./run_experiment_all.sh  # no extra arguments
```

## Visualization of experiment results

Execute the [show_aaf_all.sh](./show_aaf_all.sh) script to visualize all the
adaptive activation functions in an automated way. For parallel execution -
consult the list of commands from this script and execute them manually.

Example:

```shell
user@host:~/repo_path$ ./show_aaf_all.sh  # no extra arguments
```

## References

[^1]: Chollet, F., et al. (2015) Train a simple deep CNN on the CIFAR10 small
      images dataset. https://github.com/keras-team/keras/blob/1.2.2/examples/cifar10_cnn.py

[^2]: Krizhevsky, A. (2009) Learning Multiple Layers of Features from Tiny
      Images. Technical Report TR-2009, University of Toronto, Toronto.

[^3]: Agarap, A. F. (2018). Deep Learning using Rectified Linear Units (ReLU).
      https://doi.org/10.48550/ARXIV.1803.08375

[^4]: Elfwing, S., Uchibe, E., & Doya, K. (2017). Sigmoid-Weighted Linear Units
      for Neural Network Function Approximation in Reinforcement Learning.
      CoRR, abs/1702.03118. Retrieved from http://arxiv.org/abs/1702.03118

[^5]: Bodyanskiy, Y., & Kostiuk, S. (2023). Learnable Extended Activation
      Function for Deep Neural Networks. International Journal of Computing,
      22(3), 311-318. https://doi.org/10.47839/ijc.22.3.3225

[^6]: Bodyanskiy, Y., & Kostiuk, S. (2022). Adaptive hybrid activation function
      for deep neural networks. In System research and information technologies
      (Issue 1, pp. 87–96). Kyiv Politechnic Institute.
      https://doi.org/10.20535/srit.2308-8893.2022.1.07 

[^7]: Bodyanskiy, Y., & Kostiuk, S. (2022). Deep neural network based on
      F-neurons and its learning. Research Square Platform LLC.
      https://doi.org/10.21203/rs.3.rs-2032768/v1 
