-Configure your Hadoop cluster 
-There is a notebook named treat_data.ipynb in the notebooks directory; you can use it to crop your file into multiple subset depending on the number of processors you want to use 
-Put the training files inside HDFS file system and change the file and change the destination of files inside the training function (sorry we forgot to add facilities for this)
-You have the folder runs containing record on your future trains(to visualize it you can install pytorch tensorboard and launch on the runs folder)
- You can launch the code using the command:
-Unzip the folder finbert_single_machine and launch the notebook on the notebook folder if you want to test machine learning with no parallelism

-For the distributed machine learning do as following :
create the virtual environment

conda env create -f environment.yml
conda activate finbert
(if you have a cpu only you can need : pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html)


install cmake : https://vitux.com/how-to-install-cmake-on-ubuntu-18-04/

wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz
tar -zxvf cmake-3.15.2.tar.gz
cd cmake-3.15.2
./bootstrap
make
sudo make install
cmake --version

install g++ : sudo apt install g++-10 if you already have it in your ppa 
if not:
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-10
sudo apt install g++-10

install OpenMPI:
sudo apt-get update -y
sudo apt-get install -y openmpi-bin

install horovod:
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_SPARK=1 HOROVOD_WITH_GLOO=1 pip3 install horovod[pytorch,spark,tensorflow] --no-cache

HDFS: 
put your crop data in HDFS : $HADOOP_HOME/bin/hadoop fs -put 1/* directory_name/nbr_of_process/name_of_the_file
You can modify it directly inside the get_data function of the finbert file (put it in such a way each worker can easily identify its file)

hdfs --daemon start datanode  

time horovodrun --timeline-filename hrvd.json --gloo -np 4 -H your_hostname_IP_1:2  your_hostname_IP_2:2  --verbose python3 good_finbert_training.py nbr_of_epoch size_of_the_dataset_you_want_to_train

-----You can change the nomber of processors , here by default we put 4

---we thank the prosusAI team without whom this project could have been more difficiult for us -------