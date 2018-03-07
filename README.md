# deep-attention
This repository are mainly borrowed from fabulous neuraltalk2, hats off to Karpathy, what a great job he has done! The model implemented here is an open source code of the deep language model with attention.
Requirements
For evaluation only
This code is written in Lua and requires Torch. If you're on Ubuntu, installing Torch in your home directory may look something like:

$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using LuaRocks (which already came with the Torch install). In particular:

$ luarocks install nn
$ luarocks install nngraph 
$ luarocks install image 
We're also going to need the cjson library so that we can load/save json files. Follow their download link and then look under their section 2.4 for easy luarocks install.

If you'd like to run on an NVIDIA GPU using CUDA (which you really, really want to if you're training a model, since we're using a VGGNet), you'll of course need a GPU, and you will have to install the CUDA Toolkit. Then get the cutorch and cunn packages:

$ luarocks install cutorch
$ luarocks install cunn
If you'd like to use the cudnn backend (the pretrained checkpoint does), you also have to install cudnn. First follow the link to NVIDIA website, register with them and download the cudnn library. Then make sure you adjust your LD_LIBRARY_PATH to point to the lib64 folder that contains the library (e.g. libcudnn.so.7.0.64). Then git clone the cudnn.torch repo, cd inside and do luarocks make cudnn-scm-1.rockspec to build the Torch bindings.

For training
If you'd like to train your models you will need loadcaffe, since we are using the VGGNet. First, make sure you follow their instructions to install protobuf and everything else (e.g. sudo apt-get install libprotobuf-dev protobuf-compiler), and then install via luarocks:

luarocks install loadcaffe
Finally, you will also need to install torch-hdf5, and h5py, since we will be using hdf5 files to store the preprocessed data.

Phew! Quite a few dependencies, sorry no easy way around it :\

I'd like to distribute my GPU trained checkpoints for CPU
Use the script convert_checkpoint_gpu_to_cpu.lua to convert your GPU checkpoints to be usable on CPU. See inline documentation for why this separate script is needed. For example:

th convert_checkpoint_gpu_to_cpu.lua gpu_checkpoint.t7
write the file gpu_checkpoint.t7_cpu.t7, which you can now run with -gpuid -1 in the eval script.

License
BSD License.

Acknowledgements
Parts of this code were written in collaboration with my labmate Justin Johnson.

I'm very grateful for NVIDIA's support in providing GPUs that made this work possible.

I'm also very grateful to the maintainers of Torch for maintaining a wonderful deep learning library.
