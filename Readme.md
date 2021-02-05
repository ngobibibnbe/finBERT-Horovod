 - Unzip the folder finbert_single_machine and launch the notebook inside the notebook folder if you want to test machine learning with no parallelism .This code is the code of  ProsusAI team we modify some parameters to speedup the training while loosing a bit in term of accuracy.

   **For the distributed machine learning do as following**

- Configure your Hadoop cluster (you can  more details [here](https://www.linode.com/docs/guides/how-to-install-and-set-up-hadoop-cluster/))

- There is a notebook named `treat_data.ipynb` in the notebooks directory; use it to crop your file into multiple subsets depending on the number of processors you want to use 
- Put the training files inside HDFS file system and change the file destination (and nomenclature for all processors) inside the training `get_data` function of the FinBERT class in `finbert.py` file 

- Distribute your code to other machines at the same location

- Setup the environement 

create the virtual environment
```bash
conda env create -f environment.yml
conda activate finbert
```
If you only have a cpu you can need : 
```bash
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
- install cmake : https://vitux.com/how-to-install-cmake-on-ubuntu-18-04/
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz
tar -zxvf cmake-3.15.2.tar.gz
cd cmake-3.15.2
./bootstrap
make
sudo make install
```

- install g++ : ```bash sudo apt install g++-10 ``` if you already have it in your ppa 
if not:
```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-10
sudo apt install g++-10
```

- install OpenMPI:
```bash
sudo apt-get update -y
sudo apt-get install -y openmpi-bin
```

-  install horovod:
```bash
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_SPARK=1 HOROVOD_WITH_GLOO=1 pip3 install horovod[pytorch,spark,tensorflow] --no-cache
```
- HDFS: 
put your crop data in HDFS : ```bash $HADOOP_HOME/bin/hadoop fs -put 1/* directory_name/nbr_of_process/name_of_the_file```
You can modify it directly inside the get_data function of the finbert file (put it in such a way each worker can easily identify its file)
```bash
hdfs --daemon start datanode  
```
- Run the code
```bash 
time horovodrun --timeline-filename hrvd.json --gloo -np 4 -H host_ip_1:2,host_ip_2:2 python3 good_finbert_training.py num_of_epoch dataset_chunk_size
```
**You can change the nomber of processors , here by default we put 4**
-**we thank the @prosusAI team without whom this project could have been more difficiult for us**
