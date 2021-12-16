module purge
module load 2019
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load OpenMPI/4.0.3-GCC-9.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module unload GCCcore
module unload ncurses
module load CMake/3.11.4-GCCcore-8.3.0

PROJECT_FOLDER=$PWD

VIRTENV=hooknet_tf2.2_hvd
VIRTENV_ROOT=~/.virtualenvs

deactivate
conda deactivate

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
  yes | rm -r $VIRTENV_ROOT/$VIRTENV
  python3 -m venv $VIRTENV_ROOT/$VIRTENV
  cd $HOME
  virtualenv $VIRTENV_ROOT/$VIRTENV
  cd $VIRTENV_ROOT/$VIRTENV

  wget https://github.com/uclouvain/openjpeg/archive/v2.3.1.tar.gz
  wget http://download.osgeo.org/libtiff/tiff-4.0.10.tar.gz
  wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz

  export PATH=$VIRTENV_ROOT/$VIRTENV/bin:$PATH
  export LD_LIBRARY_PATH=$VIRTENV_ROOT/$VIRTENV/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$VIRTENV_ROOT/$VIRTENV/lib:$LD_LIBRARY_PATH
  export CPATH=$VIRTENV_ROOT/$VIRTENV/include:$CPATH

  # Installing libtiff
  tar -xvf tiff-4.0.10.tar.gz
  cd $VIRTENV_ROOT/$VIRTENV/tiff-4.0.10
  CC=gcc CXX=g++ ./configure --prefix=$VIRTENV_ROOT/$VIRTENV
  make -j 8
  make install
  cd $VIRTENV_ROOT/$VIRTENV

  # Installing openjpeg
  echo "Now installing openjpeg"
  tar -xvf v2.3.1.tar.gz
  cd $VIRTENV_ROOT/$VIRTENV/openjpeg-2.3.1
  CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$VIRTENV_ROOT/$VIRTENV -DBUILD_THIRDPARTY:bool=on
  make -j 8
  make install
  cd $VIRTENV_ROOT/$VIRTENV

  # Installing Openslide
  echo "Now installing openslide"
  tar -xvf openslide-3.4.1.tar.gz
  cd $VIRTENV_ROOT/$VIRTENV/openslide-3.4.1
  CC=gcc CXX=g++ PKG_CONFIG_PATH=$VIRTENV_ROOT/$VIRTENV/lib/pkgconfig ./configure --prefix=$VIRTENV_ROOT/$VIRTENV
  make -j 8
  make install

fi

source $VIRTENV_ROOT/$VIRTENV/bin/activate

export PATH=$VIRTENV_ROOT/$VIRTENV/bin:$PATH
export LD_LIBRARY_PATH=$VIRTENV_ROOT/$VIRTENV/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VIRTENV_ROOT/$VIRTENV/lib:$LD_LIBRARY_PATH
export CPATH=$VIRTENV_ROOT/$VIRTENV/include:$CPATH

export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
#export HOROVOD_GPU_ALLGATHER=MPI
#export HOROVOD_GPU_BROADCAST=MPI

# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  pip install --upgrade pip
  pip3 install wheel --no-cache-dir
  pip3 install openslide-python --no-cache-dir
  pip3 install setuptools --no-cache-dir
  pip3 install matplotlib --no-cache-dir
  pip3 install mpi4py --no-cache-dir
  pip3 install tqdm --no-cache-dir
  pip3 install shapely --no-cache-dir
  pip3 install opencv-python --no-cache-dir
  pip3 install tensorflow-gpu==2.3.0 --no-cache-dir
  pip3 install horovod --no-cache-dir
fi

cd $PROJECT_FOLDER
