# Privacy-Preserving Multi-label Classification for Large Language Models via Fully Homomorphic Encryption

This version extends to multi-label classification (77% - 10% test dataset - 'dair-ai/emotion').
This repo has included the notebook of fine-tuned model ('gokuls/BERT-tiny-emotion-intent')'s weights & biases extraction, and precomputed variables accumulation & extraction.
The code is built upon the implementation of the FHE circuit of the Rovida and Leporati (2024)'s work (Transformer-based Language Models and Homomorphic Encryption: an intersection with BERT-tiny).

Tony Ma

<center>
<img src="imgs/console.png" alt="Console presentation image" width=90% >
</center>

This repository contains the source code for the paper called *Transformer-based Language Models and Homomorphic Encryption: an intersection with BERT-tiny*, available at https://dl.acm.org/doi/10.1145/3643651.3659893

In particular, in contains a FHE-based circuit that implements the Transformer Encoder layers of BERT-tiny (available [here](https://huggingface.co/philschmid/tiny-bert-sst2-distilled)), fine-tuned on the SST-2 dataset.

## Prerequisites

Linux or Mac operative system

In order to run the program, you need to install:
- `cmake`
- `g++` or `clang`
- `OpenFHE` ([how to install OpenFHE](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html))

Plus, since the tokenization process (done by the client) relies on PyTorch:
- `python`
- `pip`

## How to use it
After intalling all the required prerequisites, install the required Python libraries using pip:
```
pip install -r src/Python/requirements.txt
```

Then, it is possible to generate the set of keys for the CKKS scheme. Go to the `build` folder:

```
cd build
```

and run the following command:

```
./FHE-BERT-Tiny --generate_keys
```

This generates the required keys to evaluate the circuit. Optionally, it is possible to generate keys satisfying $\lambda = 128$ bits of security by adding the following flag (notice that this will generate a larger ring, leading to larger runtimes).

```
./FHE-BERT-Tiny --generate_keys --secure
```

This command will generate a `keys` folder in the root of the project folder, containing the serializations of the required keys. Now it is possible to run the FHE circuit by using this command

```
./FHE-BERT-Tiny "Dune part 2 was a mesmerizing experience, movie of the year?"
```

In general, the circuit can be evaluated as follows (after the generation of the keys):

```
./FHE-BERT-Tiny <text> [OPTIONS]
```
where

- `<text>` is the input text to be evaluated

and the optional `[OPTIONS]` parameters are:

- `--verbose` prints information during the evaluation of the network. It can be useful to study the precision of the circuit at the end of each layer
- `--plain` adds the result of the plain circuit at the end of the FHE evaluation

## Architecture

The circuit is built to be run by a honest-but-curious server, and it is evaluated according to the following high-level architecture:

<img src="imgs/architecture.png" alt="Console presentation image" width=60% >

Find more details on the paper.

## Citing

If you are planning to cite this work, feel free to do using the following BibTeX entry:

```
@inproceedings{10.1145/3643651.3659893,
  author = {Rovida, Lorenzo and Leporati, Alberto},
  title = {Transformer-based Language Models and Homomorphic Encryption: An Intersection with BERT-tiny},
  year = {2024},
  isbn = {9798400705564},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3643651.3659893},
  doi = {10.1145/3643651.3659893},
  booktitle = {Proceedings of the 10th ACM International Workshop on Security and Privacy Analytics},
  pages = {3–13},
  numpages = {11},
  keywords = {homomorphic encryption, natural language processing, secure machine learning},
  location = {Porto, Portugal},
  series = {IWSPA '24}
}
```

## Origin FHE Circuit Authors

- Lorenzo Rovida (`lorenzo.rovida@unimib.it`)
- Alberto Leporati (`alberto.leporati@unimib.it`)

### Declaration

This is a proof of concept and, even though parameters are created with $\lambda \geq 128$ security bits (according to [Homomorphic Encryption Standards](https://homomorphicencryption.org/standard)), this circuit is intended for educational purposes only.
