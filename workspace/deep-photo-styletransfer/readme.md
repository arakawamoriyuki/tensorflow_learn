
[deep-photo-styletransfer](https://github.com/luanfujun/deep-photo-styletransfer)
[torch](http://torch.ch/docs/getting-started.html#_)


- torch install
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```

- deep-photo-styletransfer run
```
$ git clone https://github.com/luanfujun/deep-photo-styletransfer

$ cd deep-photo-styletransfer/models
$ curl -O https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt
$ curl -O http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
$ cd ..

$ luarocks install loadcaffe
$ luarocks install libcuda_utils

-gpu -1

```

th neuralstyle_seg.lua -content_image examples/high_res/1/input.png -style_image examples/high_res/1/style.png -gpu -1

```
$ th neuralstyle_seg.lua \
  -content_image examples/high_res/1/input.png \
  -style_image examples/high_res/1/style.png
  -content_seg <inputMask>
  -style_seg <styleMask>
  -index <id>
  -serial <intermediate_folder>
```