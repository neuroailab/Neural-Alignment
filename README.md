# Two Routes to Scalable Credit Assignment without Weight Symmetry

**Daniel Kunin\*, Aran Nayebi\*, Javier Sagastuy-Brena\*, Surya Ganguli, Jon Bloom, Daniel L. K. Yamins**

[Preprint](https://arxiv.org/abs/2003.01513)

## Getting started

### Installation prerequisites

- All the code is written for and tested with Python 2.7 since that is what was used for development.
- We used TensorFlow 1.13.2 for all our experiments.
- We rely on functionality implemented in the `tfutils` library for our training, which in turn requires a Mongo Database to be set up and running.

### Environment setup and requirements

We recommend setting up  a new Python 2.7 virtual environment and then install `tfutils` inside it, which will install all dependencies.

### Installing tfutils

Start by cloning [tfutils](https://github.com/neuroailab/tfutils).

```
# Https
git clone https://github.com/neuroailab/tfutils.git

# Or using SSH
git clone git@github.com:neuroailab/tfutils.git
```

** If you want to run on GPU, follow this step. Otherwise you can skip it **

Navigate into the tfutils directory, and open the file `setup.py`.

```
cd tfutils
vim setup.py
```

Edit line 78 (or the line with the `install_requires` list), to require `tensorflow-gpu<1.14` instead of `tensorflow<1.14`.
Save and close the file.

Now the install, finally. Run the setup script

```
python setup.py install
```

### Setting up a mongo database

#### Installing mongo

The `pymongo` client should have been installed by `tfutils` as a dependency.
Now you need to make sure you have mongoDB installed locally.

You can follow the instructions in [https://docs.mongodb.com/manual/installation/].

#### Running the mongo daemon

Pick a directory where the data will be stored.
We'll assume it is the home folder here.

Copy the `mongodb.conf` file to that directory.
You might want to edit some of the parameters, but the conf file as provided is set to run with the configuration tfutils requires and the port coincides with the port we specify in the training script.

```
cp mongodb.conf ~/mongo
cd ~/mongo
sudo chmod 777 mongodb.conf
```

You might also want to edit the dbpath and the logpath, in the conf file to point it to the location you chose for your MongoDB data.

Start the daemon!

```
sudo mongod --config=mongodb.conf --quiet
```

Make sure it is running by pinging it:

```
curl localhost:29029
```

If it is running correctly you should see the message:

```
It looks like you are trying to access MongoDB over HTTP on the native driver port.
```

## Training script

Starting training (in general) is as simple as running

```
python train.py
```

The script takes in several flags, the documentation for what each flag does can be found either inline in the code, or can be accessed from the command line via.

```
python train.py --help
```

or

```
python train.py --helpfull
```

Note, however, that if you choose to restore one of our provided configuration files, most of the flags described above will be ignored, as they have already been set to appropriate values to reproduce our experiments.

## Reproducing our reported experiments

First, download our config files by running the provided `config/get_configs.sh` script from within the `config` directory.

To train using the provided configs, simply use the following flags, which are required:
- `--data_dir`: The directory where ImageNet is stored in tfrecords format. This could be a gs bucket, if you wish to train on TPU.
- `--load_params_file`: The location of the config file to load from.

We provide some additional flags to configure the run according to your environment:
- `--gpu`: a comma separated list of the indices of the gpu devices to use.
- `--dbname`: The name of the MongoDB database to save to. This can be a new or existing database name in the MongoDB instance set up above.
- `--port`: The port where the mongod daemon is running. The script attempts to connect to `localhost:<port>`, so if the database is running remotely, make sure port forwarding is set up correctly.
- `--exp_id_suffix`: A string with any identifier you might want to append to the `exp_id` with which data will be saved to the DB.
- `--cache_dir`: A local directory where you wish to save the tfutils caches to. It defaults to `~/.tfutils`.
- `--model`: The name of the model to train as a string. You can choose between resnet18, resnet34, resnet50, resnet101 or resnet152.
- `--use_resnet_v2`: A boolean on whether to use the the preactivation (v2) or the post-activation form (default) of Residual Networks. Using v2 will do inception preprocessing on the data and use 299x299 sized images.

If training on TPU, you need to specify these flags instead of the `--gpu` flag:
- `--tpu_name`: The name of the TPU device you wish to train on. To simplify the configuration, we only support v2-8 and v3-8 devices.
- `--tpu_zone`: The zone in which the TPU device is located.
- `--gcp_project`: The identifier of the GCP project which hosts your instance and TPU device.

Full example to train symmetric alignment on 2 GPUs using a bigger model and custom database:

```
python train.py --gpu=6,7 --dbname=reproduction --port=29029 --cache_dir=~/tfutils_caches --exp_id_suffix=_test0 --data_dir=/mnt/fs4/Dataset/TFRecord_Imagenet_standard/image_label_full/ --model=resnet101 --use_resnet_v2=True --load_params_file=config/symmetric.pkl
```

## Loading and validating on our provided checkpoints

Additionally, we provided pre-trained checkpoints of our models, so the training process can me avoided.

First, download the checkpoint files by running the provided `checkpoints/get_checkpoints.sh` script from within the `checkpoints` directory as well as our configuration files by running the provided `config/get_configs.sh` script from within the `config` directory.

To validate using the provided checkpoints, simply use the following flags, which are required:
- `--data_dir`: The directory where ImageNet is stored in tfrecords format. This could be a gs bucket, if you wish to train on TPU.
- `--load_checkpoint`: The location of the checkpoint file to load from.
- `--load_params_file`: The location of the config file to load from.
-- `--val_exp_id`: a new exp_id used to store the result of the validation
-- `--model`: The model, as above, that corresponds to the checkpoint you are loading from

Full example to validate Resnet18 on the provided symmetric alignment checkpoint:

```
python test.py validate --gpu=6 --dbname=validate --port=29029 --model=resnet18 --data_dir=/mnt/fs4/Dataset/TFRecord_Imagenet_standard/image_label_full/ --val_exp_id=validation --load_params_file=config/symmetric.pkl --load_checkpoint=checkpoints/resnet18/symmetric/model.ckpt-450360
```

## Cite

```
@article{kunin2020routes,
    title={Two Routes to Scalable Credit Assignment without Weight Symmetry},
    author={Daniel Kunin and Aran Nayebi and Javier Sagastuy-Brena and Surya Ganguli and Jon Bloom and Daniel L. K. Yamins},
    year={2020},
    eprint={2003.01513},
    archivePrefix={arXiv},
    primaryClass={q-bio.NC}
}
```
