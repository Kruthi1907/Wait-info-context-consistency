# Wait-Info with Context-Consistency-Bi Training

# Steps to Implement this Strategy

## Installing Requirements
```bash
# First, install Python 3.9
sudo apt-get update
sudo apt-get install python3.9
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
echo 1 | sudo update-alternatives --config python3
sudo apt-get install python3.10-distutils
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py

pip install --upgrade setuptools==65.5.0 pip==23.0.1
pip install PyYAML sacremoses subword-nmt sacrebleu

pip install tensorboardX
```

```bash
pip install --editable ./
```

## Data preparation:

```bash
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..
```

## Data preprocessing:

```bash
TEXT='examples/translation/iwslt14.tokenized.de-en'
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

# Training:

```bash
fairseq-train /content/fairseq/data-bin/iwslt14.tokenized.de-en \
    --arch transformer_wait_info_arch \
    --optimizer adam \
    --lr 0.001 \
    --update-freq 16 \
    --save-dir checkpoints/transformer_wait_info \
    --batch-size 32 \
    --max-epoch 50 \
    --max-source-positions 1024 \
    --max-target-positions 1024
```

# Testing:

```bash
fairseq-generate \
    data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/transformer_wait_info/checkpoint_best.pt \
    --batch-size 128 \
    --beam 5 \
    --remove-bpe \
    --user-dir /content/fairseq/fairseq/models
```
