# Native Language CNN

CNN-based and grammar-rules based model for native language identification based on speech transcripts.

## Getting Started

These instructions will get you a copy of the project up and running on your machine for development purposes.

### Prerequisites

Create an environment with the required Python packages using your favorite package manager.

```
conda create --name <env> --file requirements.txt
```

## Training

To train a CNN model, run

```
python code/train.py
```

To do so with GPU support,

```
python code/train.py --cuda 0
```

see `train.py` help for more information.

## Testing

Refer to `Evaluate.ipynb` notebook for information on model evaluation.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

This work is derived from data provided by the Educational Testing Services, copyright 2017 ETS (www.ets.org). The opinions set forth in this work are those of the authors and not ETS.
