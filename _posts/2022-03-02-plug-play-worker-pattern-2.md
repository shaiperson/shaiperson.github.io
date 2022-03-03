---
layout: post
title: "Plug & Play Worker Pattern - Part II"
date: 2022-03-02 12:55:00 -0300
tags:
  - ml-ops
  - tech
---
---
**Note**

This is **Part II** of a three-part series.

- Part I - use case, pattern concepts
- Part II - naïve implementation
- Part III - pattern implementation

---

## Introduction

Let's quickly review Part I of the series.

In Part I, we:
- Made the case for extensibility as a quality attribute to seek in designing algorithm-centric production workflows.
- Presented a fictional use case for a containerized analysis worker environment, a common kind of setup in real-world applications, that reads images from a queue and runs algorithms on them.
- Came up with an initial design the worker, naïve with respect to extensibility.
- Analyzed what made the initial design less extensible and used our conclusions to come up with a design pattern that solves for extensibility. 

In Part II, we'll go through implementing the naïve design. This is useful mostly as a precursor to Part III where we'll re-implement our worker using the final design pattern. By first having the initial design coded and functioning, we'll be able to apply the pattern to it, test that it still works and look at how much easier it is to extend in practical terms.

### A note on implementation and code structure

As explained in Part I, we'll implement everything in Python. I include some code snippets throughout the article tailored to aid discussion, but you can find the complete working code for the example [this GitHub repo](https://github.com/shaiperson/worker-pattern-article). Code for the initial design presented here is in branch `initial`. 

The code for all componentes is placed in a single repository. Looking at the repo, you'll find a directory for each component with its source files, Dockerfile and a `build.sh ` script. The `worker` and `producer` directories are there to assist in running and testing everything locally.

## Implementation

To quickly review Part I, the initial design we'll implement here consists of a Controller component that reads images from a queue and sends them to a Runner component. The latter houses the actual algorithm code, and exposes it on an auxiliary HTTP server for the controller to send requests to.

<img src="/assets/plug-play-worker-pattern-part-1/Untitled.png" style="display: block; margin-left: auto; margin-right: auto; width: 30%;"/>

### Controller

We'll be using a RabbitMQ queue called `tasks` in our example setup, so we set up our controller with `pika` to consume from the `tasks` queue.

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='tasks')
```

We define a callback to be run on each consumed message. This callback does the controller work: call the runner, check for errors in the response and do something with the result or the error. In our example, we just log the result if the image was processed successfully and the error otherwise.

```python
import requests

RUNNER_HOST = os.environ.get('RUNNER_HOST', 'localhost')
RUNNER_PORT = os.environ.get('RUNNER_PORT', 5000)

def callback(c, m, p, body):
		print('Received message, calling runner')
    headers = {'Content-Type': 'application/json'}
    url = 'http://{}:{}'.format(RUNNER_HOST, RUNNER_PORT)
    response = requests.request("POST", url, headers=headers, data=body)
    if response.ok:
        print(f'Received result from runner: {response.json()}')
    else:
        print(f'Received error response from runner: {response.status_code} {response.json()}')
        # Handle error (retry/requeue/send to dead-letter exchange)
```

Lastly, we start listening for messages on the queue.

```python
channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)
print(f'Listening for messages on queue {QUEUE_NAME}')
channel.start_consuming()
```

### Classifier

I trained a simple classifier for the purposes of this article. Since our focus here is on the ML Ops side of things, we'll just use the model as a black box and look at how to use and deploy it. However, you can check out the supported memes, model, data and full code for training in this [kaggle notebook](https://www.kaggle.com/shaibianchi/meme-classifier/). Credit to Keras's [transfer learning guide](https://keras.io/guides/transfer_learning/) and to [gmor's meme dataset on Kaggle](https://www.kaggle.com/gmorinan/memes-classified-and-labelled).

We define a `classifier` module responsible for loading and running the classifier. In this module, the model is loaded and compiled from its `.h5` serialization upon initialization. In this case, we assume the `.h5` file is packaged with the code in a `model/` directory at root level.

```python
import tensorflow as tf

with open('model/meme-classifier-model.json') as f:
    model = tf.keras.models.model_from_json(f.read())

model.load_weights('model/meme-classifier-model.h5')
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
```

For running the model, we expose a `run_on_url` function. It uses some auxiliary functions to process the input image and the output. We'll omit the code for these functions to reduce cluttering here, but you can find it in the repo.

```python
def run_on_url(url):
    logger.debug('Fetching image')
    image_bytes = _get_image_bytes(url)

    logger.debug('Reading and preparing image')
    image_tensor = _get_image_tensor(image_bytes)

    logger.debug('Running on image')
    pred = model.predict(image_tensor)

    return _pred_to_label(pred)
```

We also define a `server` module responsible for exposing the classifier on a local HTTP endpoint. We use the Python `http` module to set up a simple local HTTP server that calls `classifier.run_on_url` upon `POST` requests on `/`  with a JSON body containing an image URL.

```python
class ClassifierServer(BaseHTTPRequestHandler):
    def _set_response(self, code, content_type):
        self.send_response(code)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_POST(self):
        # Process request
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            post_json = json.loads(post_data)
            image_url = post_json['url']

        except ValueError as e:
            logger.error('ValueError while parsing body as JSON')
            response_body = {'message': f'Task body not valid JSON: "{e}"'}
            self._set_response(400, 'application/json')

        except KeyError as e:
            response_body = {'message': f'Task body missing field {e}'}
            self._set_response(400, 'application/json')

        else:
            try:
                logger.debug('Running algorithm on URL {}'.format(image_url))
                result = classifier.run_on_url(image_url)
                response_body = {'result': result}
                self._set_response(200, 'application/json')

            except exceptions.RequestError as e:
                response_body = {'message': f'Error fetching request image, received {e.response.status_code}'}
                self._set_response(e.response.status_code, 'application/json')

            except Exception as e:
                response_body = {'message': 'Internal error', 'error': error_str}
                self._set_response(500, 'application/json')

        self.wfile.write(json.dumps(response_body).encode('utf-8'))
```

Notice the error management. As we saw above, the controller looks out for an error status being returned by its runner and does some management of the failure to process a given message (like logging it and/or sending it to a dead-letter exchange). So in implementing the runner, we want to be diligent in capturing expected errors so that we can return meaningful error status for the controller to handle, which will later help us quickly understand failed messages and be robust in managing unexpected errors.

### Dockerization, Compose File

I tend to favor containerizing all componentes that go into a worker setup as this lends itself very well to deployment using orchestration tools such as AWS ECS or Kubernetes. It also helps in minimizing differences between development, staging and production environments. We'll use Docker Compose to run our setup locally.

#### Controller Dockerfile

```dockerfile
FROM python:3.7

WORKDIR /opt/project

COPY ./* ./

RUN pip install -r requirements.txt
```

#### Classifier Dockerfile

```dockerfile
FROM tensorflow/tensorflow

WORKDIR /opt/project

RUN pip install pillow

COPY ./* ./
COPY ./model ./model
```

#### Compose file

We set our environment up with the queue service and both components of our worker. I only show some relevant fields from the file, but you can check it out with all of its details in the repo.

The `restart` and `depends_on` clauses in the `controller` service are there to allow a warmup period for the `rabbitmq` service after it starts.

```yaml
version: '3'
services:
  rabbitmq:
    image: rabbitmq:3 
    ports:
      - 5672:5672
      
  meme-classifier:
    image: meme-classifier
    command: python server.py
    
  controller:
    image: controller
    command: python main.py
    restart: on-failure
    depends_on:
      - rabbitmq
```

#### Producer

The code in the `producer` directory is there to aid in testing. It's set up to connect to the queue and allow us to easily send some test messages for our worker to process.

### Testing it out

After building the component images and tagging them appropriately, run `docker-compose up -d` in the `worker ` directory to spin up the environment. Run `docker-compse logs -f` to track initialization. You're likely to see some connection errors from `controller` as it fails to connect to `queue` while the latter completes its initialization.

Once both the controller and classifier services are listening for messages and requests respectively, we can send some meme image URLs to the queue and get some classification happening.

```
...
meme-classifier | INFO :: [+] Listening on port 5000
...
controller      | INFO :: [+] Listening for messages on queue tasks
```

Let's try one out.

![What if I told you / Matrix Morpheus - what if i told you i have no idea how my glasses dont fall out](https://memegenerator.net/img/instances/39673831.jpg)

From a terminal at `./producer/`:

```bash
$ python main.py "https://memegenerator.net/img/instances/39673831.jpg"
```

Logs:

```
controller         | INFO :: Received message, calling runner
meme-classifier    | INFO :: Running classifier on URL
controller         | INFO :: Received result from runner: {'label': 'matrix_morpheus', 'score': 0.99989}
```

Looking good! You can play around some more with it if you like and have some fun looking at memes as I have while doing the same 😛.

## What's Next

In Part III of the series, we'll implement the final pattern presented in Part I and test it. We'll then try to further extend our resulting setup with a new algorithm and see in practical terms if we achieved our goal of making it low-overhead and easy.
