# Core

Core contains important primitives for the encoder nodes


## Encoder Node

Details how the encoder node assembles itself given some inputs. The encoder node is used to take many independent modules and put them together in a cohesive way as well as guaranteeing that each module starts and shuts down correctly.


## Internal Broadcaster

The logic for broadcasting to other encoders along with handling retries and a backoff scheme. 


## Pipeline Dispatcher

A central dispatcher that sends to different pipelines
