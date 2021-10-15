# Controller

Fast API interface to access the worker and orchestrate training

## Start Train

-   On recieving an HTTP/REST request to start training controller produces message in `train` topic.


## Directory Structure

```
controller/
├── app/
│   ├── core/
│   ├── __init__.py
│   └── main.py
├── Readme.md
└── tests/
```
