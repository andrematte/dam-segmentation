# Docker Commands

```sh
docker build . -t dam_segmentation:0.1.0  
docker run --rm -it -v ./data:/app/data damseg_rf:0.1.0 python scripts/create_sample_data.py
```
