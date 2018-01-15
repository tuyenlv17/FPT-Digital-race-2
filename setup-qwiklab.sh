git config --global alias.ss 'status -s'
git config --global credential.helper "cache --timeout=36000"
git config --global user.email "blabla@nvidia.lab"
git config --global user.name "blabla-nvidia"
pip install --user tensorflow-gpu
pip install --user pandas
pip install --user sklearn
pip install --user scikit-image==0.13.1
pip install --user nolearn
pip install --user matplotlib
pip install --user -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
pip install --user Lasagne==0.1
pip install --user cloudlog

#setup
mkdir -p tools-bin/cuda/lib64
mkdir -p tools-bin/cuda/include
mkdir -p tools-bin/cudnn
cat tools/cudnn/cudnn-part* > tools-bin/cudnn/cudnn-8.0-linux-x64-v6.0.tgz
tar xvf tools-bin/cudnn/cudnn-8.0-linux-x64-v6.0.tgz -C tools-bin/cudnn
CUR_DIR=$(pwd)
CUDA_DIR="$CUR_DIR/tools-bin/cuda"
echo "install cuda to this dir $CUDA_DIR\n"
# python download-cuda.py
# bash ./tools-bin/cuda_9.1.85_387.26_linux.run
mv tools-bin/cudnn/cuda/include/cudnn.h tools-bin/cuda/include/
mv tools-bin/cudnn/cuda/lib64/* tools-bin/cuda/lib64/
# env
echo "add to profile export PATH=\"$CUDA_DIR/bin:\$PATH\""
echo "export PATH=\"$CUDA_DIR/bin:\$PATH\"" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\"$CUR_DIR/tools-bin/cuda/lib64:/usr/local/cuda-8.0/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
echo "export PATH=\"$CUDA_DIR/bin:\$PATH\"" >> ~/.profile
echo "export LD_LIBRARY_PATH=\"$CUR_DIR/tools-bin/cuda/lib64:/usr/local/cuda-8.0/lib64:\$LD_LIBRARY_PATH\"" >> ~/.profile
. ~/.bashrc
cat traffic-signs-data/train-split/train.pa* > traffic-signs-data/train.p