# SAM2 API Server

Provides a REST API for the SAM2 system to process a designated video to label objects.


## Deploy Build

Installed in
~/Python/SegmentAnything


```
cd ~/Python/SegmentAnything
git clone https://github.com/facebookresearch/sam2.git && cd sam2
```

```
cd ~/Python/SegmentAnything
cd sam2
python3.11 -m venv myenv
source myenv/bin/activate
```


## Install Packages

```
pip install -e .
```

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

```
pip install -e ".[notebooks]"
```

```
jupyter notebook
```



## Run API Server

Make sure Venv is running:
```
cd ~/Python/SegmentAnything/sam2
source myenv/bin/activate
```

Running flask api within Venv:

```
cd api_server/
export PYTORCH_ENABLE_MPS_FALLBACK=1
python3.11 sam_api_server.py
```

If we dont export PYTORCH_ENABLE_MPS_FALLBACK=1 for running on MacOS, then we get an error:
```
RuntimeError: MPS backend is not available.
```

Server will be running by default on http://127.0.0.1:3030.
