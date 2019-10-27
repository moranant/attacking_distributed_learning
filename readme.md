# A Little Is Enough: Circumventing Defenses For Distributed Learning

In order to see the parameters for the experiments just use `main.py -h`
We use python 3, and we build upon pytorch.

Unfortunately, the code for training CIFAR100 over GPU (CPU is too slow) was not ready to be shared. 

For backdooring (-b option), you can either use "No" backdooring, "Pattern" backdooring for changing the top-left 5*5 to the max intensity as described in the paper, or an index for the specific index of the image from the dataset to behave as a backdoor sample.

## Authors

* **Moran Baruch**
* **Gilad Baruch**
* **Yoav Goldberg**
