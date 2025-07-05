# 1. Download conda and kaggle training and test data.

https://www.anaconda.com/download

https://www.kaggle.com/competitions/digit-recognizer/overview

And place the training and test data in images_data folder

# 2. Create conda environment

```bash
conda create -n digit_rec
conda activate digit_rec
pip install -r requirements.txt
```
# 3. Train and run the model

## (Optional) Run hyper-parameter optimization script

```bash
python hpo_script.py --trials <num_of_trials>
```

If a better set of hyper-parameters emerges, change the parameters in the train_model.py

## Train the model

```bash
python train_model.py
```

## Make predictions on Kaggle test data

```bash
python predict_data.py
```

