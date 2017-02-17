
# sh classify_image.sh

export TENSORFLOW_MODELS=/Users/t01530/Documents/repositories/tensorflow_learn/models
export TENSORFLOW_IMAGENET=tutorials/image/imagenet

export IMAGE_DIR=/Users/t01530/Documents/repositories/tensorflow/workspace/classify_image/images

classify_image () {
    python $TENSORFLOW_MODELS/$TENSORFLOW_IMAGENET/classify_image.py --image_file $1
}

for file in `\find $IMAGE_DIR -maxdepth 1 -type f`; do
    classify_image $file
done