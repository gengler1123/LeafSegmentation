# Leaf Mask Data Generation

We found it during previous examples working with this data that the `leaf` and `leafscan` content types
led to the highest quality returns from the system.  This repository is designed to create COCO-type masks
to train the Mask R-CNN (either TensorFlow or PyTorch).  Leaf mask creation is designed to be the first step
of a plant identification process.

### Dataset

The primary dataset chosen for this is the LifeCLEF 2015 Plant Task https://www.imageclef.org/lifeclef/2015/plant
which is primarily based on plant species in Europe.  The link includes more detail on the 
task and the datset itself.

### Motivation
The primary motivation for this project is towards building an `invasive species` detection
system which more naturally integrates with photo apps on a persons phone for after / during
hiking identification and notification of invasive plant species.  These models are a first
step towards a more (read _actual_) system to perform that task.
