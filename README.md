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

see [train.py](code/train.py) help for more information.

## Testing

Refer to the [Evaluate.ipynb](Evaluate.ipynb) notebook for information on model evaluation.

## Acknowledgments

This work is derived from data provided by the Educational Testing Services, copyright 2017 ETS (www.ets.org). The opinions set forth in this work are those of the authors and not ETS.

## License

To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>
