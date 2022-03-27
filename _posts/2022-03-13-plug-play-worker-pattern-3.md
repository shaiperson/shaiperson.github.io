---
layout: post
title: "Plug & Play Worker Pattern - Part III"
date: 2022-03-02 12:55:00 -0300
tags:
  - ml-ops
  - tech
---
---
**Note**

This is **Part III** of a three-part series.

- Part I - use case, pattern concepts
- Part II - naïve implementation
- Part III - pattern implementation

---

## Introduction

In this article series, we develop a design pattern for worker environments that run algorithms on data. The goal of the pattern is to make the environment easily extensible with new algorithms.

In Part I we developed the theory for the pattern, starting from an initial, naïve implementation of a worker environment that does not take extensibility into account, building all the way up to a pattern that attempts to optimize for that property. To also build up to the pattern in the technical realm, we implemented the initial design in Part II to use as a basis for implementing the final design. 

Here, in Part III, we code up the final form of the pattern. We then actually go ahead and add new algorithms to the environment in different ways to see the greater extensibility in action and enjoy the fruits of our work!

## Implementation

Let's quickly review what we have to do. As we saw in Part I, the pattern in its general form looks like this:

<img src="/assets/plug-play-worker-pattern-part-1/Untitled%203.png" style="display: block; margin-left: auto; margin-right: auto; width: 50%;"/>

We have a Runner Discovery component responsible for holding a registry of supported algorithms that allows the Controller to discover them. In turn, the "Runner" components have the capacity of running algorithms. They first register with the Discovery component and then listen for algorithm requests over HTTP that the Controller can send when it's ready. As you can tell, the order of initialization clearly matters. The numbers in the diagram denote this order:

1. Runner Discovery initializes.
2. Runners initialize, POST list of supported algorithms and port to Runner Discovery.
3. Controller initializes, GETs algorithm-to-port mapping from Runner Discovery.
4. Controller sends algorithm requests to runners until work is done.

Now, let's go ahead and apply this pattern to our worker environment from the previous article. We'll start with the Runner Discovery, which is both the only brand-new element in our environment and the first one to initialize. Then, we'll look at how to adapt our runner component to the component. Last, we'll extend the controller with the necessary code to have it discover the runners automatically.

#### General note

As we did in the previous article when implementing our naïve setup, we'll Dockerize each component that goes into our worker environment and run all of them in concert using Docker Compose. To make it all work, there are a number of configuration parameters that need to be correctly set in each container of the setup, and we'll take care to make all of these configurable by environment variables. In each Python projects that requires it, we'll use the convention of creating a `settings` module that picks up all relevant environment variables, validates them and exposes them to other modules.

Similarly to the previous article, we'll implement everything here in Python. I include some code snippets throughout the article tailored to aid discussion, but you can find the complete working code for the example this GitHub repo. Code for the complete pattern presented here is available in branch `pattern`. The code for all components is placed in a single repository. Looking at the repo, you'll find a directory for each component with its source files, Dockerfile and a `build.sh` script. The `worker` and `producer` directories are there to assist in running and testing everything locally.

### Runner Discovery

This component has a simple, single responsibility: to function as a discovery service for runners and algorithms. As such, it'll be a server application exposing a simple API for registering discoverable componentes and reading them. This means it has to:

1. Provide an endpoint for runners to register themselves and the algorithms they support on.
2. Provide an endpoint for the controller to discover runners on when it needs to.

Taking from the concepts we laid out in Part I, this functionality is what endows our environment with a _dynamic mapping_ of algorithms to runners.

As in Part II, we'll leverage FastAPI to write a succinct definition of our API and use Uvicorn to run it.

The model for registration requests consists simply of an `{algorithm, host}` pair with the name of an algorithm and the URI of the runner it can be found on.
```python
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

class AlgorithmRegistrationRequest(BaseModel):
    algorithm: str
    host: str
```

The API allows runners to `POST` these `{algorithm, host}` pairs for the Discovery component to store in its registry, and to `GET` the registry so that the controller learns on which host it can find each algorithm.

```python
app = FastAPI()

registry = {}

@app.post('/')
async def register_algorithm(request: AlgorithmRegistrationRequest):
    logger.info(f'Registering algorithm {request.algorithm} as hosted on {request.host}')
    registry[request.algorithm] = request.host

@app.get('/')
async def get_registry():
    return registry
```
 
 
### Extending the runners
 
As we've seen, to adapt the runner components to this pattern they now need to notify the Runner Discovery component of the URI on which they're reachable and the algorithms they support. This is in addition to still exposing the algorithms on HTTP endpoints for the controler to send requests on and running the algorithms themselves.
 
As we hinted at in Part I when going over the pattern's theory, the only way we can really make a setup such as this easy enough to develop and maintain is to _single-source_ those environment-related responsibilities of integration with the Discovery and Controller components. In other words, we want to develop the code for these responsibilities separately from the runners themselves, and somehow package the runners with it so that they're automatically enhanced with those capabilities.
 
Working with Python, the way we'll do this is to develop a separate Python project that does all of the environment-related work. We'll then package that project as a dependency that can be installed in each runner container using `pip`. This package will be called `runnerlib`.

Also, by developing `runnerlib` separately, we can concentrate exclusively on algorithms whenever we're working on a project responsible for algorithms. For this to work, however, we have to come up with some kind of convention that will make each project's supported algorithms discoverable by `runnerlib` since, at build time, `runnerlib` will be naturally agnostic of the runner it will be packaged with in each case. We'll outline such a convention for the runners to comply with presently.
 
#### Environment-related code

**Discovery**

As a first step in developing `runnerlib`, let's create a `disocvery` module responsible for interacting with Runner Discovery. This module will expose a function, called `register_all`, that discovers the locally supported algorithms and sends registration requests for those algorithms to the Discovery component. This is where a convention for `runnerlib` and the runners to agree on becomes necessary: what can we do in our algorithm-running project to make its algorithms easy to discover by a library installed in the same Python environment?

There are many ways to go about this, but let's just go with one. This is the convention:

1. Each runner project sports a module called `runner_adapter`.
2. The `runner_adapter` module in each runner exposes one handler function per supported algorithm.
3. The name of each handler function is `run_{algorithm-name}`.
4. The arguments for each handler function are sufficient to run its associated algorithm, are specified with type hints and all types convertible from JSON.
5. The return value from each handler is sufficient in capturing the result for the algorithm run and it's convertible to JSON.

The first three requirements allow `runnerlib`'s `discovery` module to discover the algorithms and the Python functions by which it can run them on data. The fourth requirement allows it to create Pydantic models for each function to use for validation on payloads sent in algorithm-running requests from the Controller. Creating these models with Pydantic also enables easily generating documentation for them. The fifth and last requirement serves to simplify generating the server's response to each request. In this way, the `runner_adapter` serves to interface between the environment-related operations upstream from it and the logic of running and computing algorithm results downstream from it. If any conversions need to be made on what's returned from downstream algorithm code, they can be made in the adapter to yield a result that complies with this contract.

We're now ready to code the `register_all` function. The function first imports the `runner_adapter` module that should be available to import once `runnerlib` is installed in a given runner container if the convention is complied with. It then uses `inspect` to pick up all `run_{algorithm-name}` functions and map each algorithm name to its handler in the `handlers_by_algorithm` dict. Finally, it sends `POST`s each algorithm to the Discovery component along with the runner's URI available to it at `settings.host`.

```python
import inspect
import re

import requests

from .setings import settings

handlers_by_algorithm = None

def register_all():
    import runner_adapter

    algorithm_handlers = inspect.getmembers(
        runner_adapter,
        predicate=lambda f: inspect.isfunction(f) and re.match(r'run_*', f.__name__)
    )

    # Map each algorithm name to its handler locally
    global handlers_by_algorithm
    handlers_by_algorithm = {name.split('run_')[1]: function for name, function in algorithm_handlers}

    # Register each algorithm with runner discovery
    unsuccessful = []
    for name in handlers_by_algorithm:
        body = {'algorithm': algorithm_name, 'host': settings.host}
        response = requests.post(settings.runner_discovery_uri, json=body)
        response.raise_for_status()
```

In the `discovery` module we also expose a getter that returns a handler given an algorithm name:

```python
def get_handler(algorithm):
    return handlers_by_algorithm.get(algorithm, None)
```

**Dynamically generated request models**

The other environment-related responsibility `runnerlib` has to endow our runners with is to spin up a server for the Controller to hit with algorithm-running requests.

As we hinted at before, we can dynamically create Pydantic models for each handler's expected arguments by using inspection. This is done with Pydantic's `create_model` function that let's us create a model with fields and types only known at runtime. Because at runtime we know what algorithms our current runner supports, we can also create a Pydantic model to validate the requested algorithm itself is supported. This can be done very comfortably by using FastAPI and defining the algorithm as a path parameter of that Pydantic model's type.

Let's first go over the dynamic model creation, implemented in a `models` module.

First some imports, and the declaration of a variable that'll be used by the server code to get the relevant models organized by algorithm. Note that, in particular, we also import `discovery`'s `handlers_by_algorithm` both to get the set of supported algorithms and because it's by inspecting these handlers that we can tell what arguments they expect. 
```python
import inspect

from pydantic import create_model, Extra
from enum import Enum

from .discovery import handlers_by_algorithm

request_models_by_algorithm = {}
```

We loop over supported algorithms, inspect each handler and generate a Pydantic model dynamically. We also populate a list of algorithm names to use 
```
class Config:
    extra = Extra.forbid

# Dynamically create
for name, handler in handlers_by_algorithm.items():
    argspec = inspect.getfullargspec(handler)
    typed_argspec = {field: (typehint, ...) for field, typehint in argspec.annotations.items()}
    request_model = create_model(f'AlgorithmRequestModel_{name}', **typed_argspec, __config__=Config)
    request_models_by_algorithm[name] = request_model
```

Lastly, by iterating over `handlers_by_algorithm`'s keys, we can create an enumeration model of supported algorithms:
```python
SupportedAlgorithm = Enum('SupportedAlgorithm', {name: name for name in handlers_by_algorithm})
```

As a nice bonus, we can add a function in this module that returns the schemas for the generated modules. This can be used to get a quick view of the payloads expected by runners and their algorithm request handlers and create documentation.
```python
def get_schemas():
    return {name: model.schema() for name, model in request_models_by_algorithm.items()}
```

**Server**

Now, to the server itself.

Aside from server-related dependencies, we import from `models` everything we need to run validation in our API as discussed above, and `discovery`'s `get_handler` that, for each supported algorithm that's requested, will get us it's corresponding handler exposed in `runner_adapter`.

We define a single `POST` endpoint that takes the algorithm to run as a path parameter, and a body that must correspond to that algorithm's handler's arguments as defined in `runner_adapter`. FastAPI will validate that the algorithm is supported by having declared `algorithm` as a path parameter of type `SupportedAlgorithm`, and then inside our path operation function we run validation of the body against the model we generated dynamically for the requested algorithm, found in `request_models_by_algorithm` exposed by `models`. If validation passes, we just invoke the handler passing it the body's content as arguments and return the result, which will be successfully converted to JSON by FastAPI if the convention from before was followed in coding the container's `runner_adapter`. As in the previous article, `exceptions` is a local module defining expected exceptions in our application (find it in the repo).
```python
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic.error_wrappers import ValidationError

from .models import SupportedAlgorithm, request_models_by_algorithm
from .discovery import get_handler
import exceptions

app = FastAPI()

@app.post("/run/{algorithm}")
async def run_algorithm(algorithm: SupportedAlgorithm, payload: dict):
    algorithm_name = algorithm.value

    # Validate payload using dynamically generated algorithm-specific model
    try:
        request_models_by_algorithm[algorithm_name].validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    handler = get_handler(algorithm_name)

    try:
        return handler(**payload)
    except exceptions.RequestError as e:
        raise HTTPException(status_code=400, detail=f'Error fetching request image, received {e.response.status_code}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
```

Finally, we also expose a function in the `server` module that runs the server:
```python
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
```

**Top-level code**

Lastly, since all these environment-related concerns essentially make up the initialization process of a runner container, in the Docker command in each runner container we'll actually run `runnerlib` as a top-level module. This means we code the runner's initialization in `runnerlib`'s `__main__` module:

```python
from .discovery import register_all
register_all()

from .server import run_server
run_server()
```

Upon running `python -m runnerlib` in a runner container, that runner's algorithms will get registered with the Discovery component, and it'll be listening for algorithm-running requests. 

#### Complying with the convention

To comply with the convention we came up with for runners to integrate with `runnerlib`, we just need to create a `runner_adapter` module in each algorithm-running project. By following the five requirements we outlined before, we get the following very simple module code. The algorithm name being `meme_classifier`, we define a `run_meme_classifier` handler function with a JSON-friendly argument in `image_url` that is enough to run the algorithm and a result that upstream concerns can convert to JSON. This handler calls the `run_on_url` function we saw in Part II, which remains exactly the same, as well as the rest of the algorithm-running logic itself that is now encapsulated behind the adapter.

```python
import classifier

def run_meme_classifier(image_url: str):
    # logger.info('Running classifier on URL'.format(image_url))
    label, score = classifier.run_on_url(image_url)
    return {'label': label, 'score': float(f'{score:.5f}')}
```

#### Packaging runners with `runnerlib`

By installing `runnerlib` in a runner's container, it's available to run inside it as a top-level module. The command to run the container with is then simply `python -m runnerlib`. To install `runnerlib`, the Dockerfile I created in the repo simply copies the `runnerlib` code found inside the repo to the container image and runs `pip install` on it. There are many other ways to install an in-house Python package as a dependency in a container, and the best one will depend on development and CI/CD processes. Note that, in any case, `runnerlib` is single-sourced and thus can be developed in one single place, versioned separately and distributed easily to any number of runner containers using a single process.

### Extending the controller

The only bit of code missing is to extend the Controller with some logic to get the runner registry from the Discovery component. This is a very simple addition to make: by using the API we defined for Discovery RUnner, just send a `GET /` request to it and get a dictionary that maps algorithm names to local runner hosts.

```python
runner_registry = requests.request('GET', runner_discovery_uri).json()
```

If we tweak format of messages sent to the queue to include the name of an algorithm to run alongisde the data to run it on, then the algorithm name can be used to get the corresponding runner host that supports that algorithm by reading `runner_registry`. If `algorithm` is the field in the queue message's `body` that gives us that name and `payload` is the field with the data to run it on (compliant with the runner's algorithm handler arguments as defined in its `runner_adapter`), then the following bit of code gets us home:

```python
import json

import requests

algorithm = body['algorithm']
payload = body['payload']

runner_uri = f'{runner_registry[algorithm]}/run/{algorithm}'
response = requests.request('POST', runner_uri, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
```

### Dockerization, Compose File

Not much changes in the Dockerfiles we already went over in the Part II, save an addition to the runner's that installs `runnerlib`:

```dockerfile
COPY ./runnerlib /opt/lib/runnerlib
WORKDIR /opt/lib/runnerlib
RUN pip install .
```

As for our brand new Discovery component, its Dockerfile is pretty straightforward as well. The requirements are all to do with setting up its server using FastAPI.
```dockerfile
FROM python:3.7
WORKDIR /opt/project
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./* ./
```

In turn, the Compose file we use to run and test out our environment gets a new service specification for Discovery Runner and some new environment variables to serve as configuration paramteres. The important additions in the way of configurability have to do with the discovery mechanism we designed. We now have to pass Runner Discovery's URI to the Controller and runner so that they can communicate with it, and we have to make the runner container know its container name so it can pass it to Runner Discovery when it registers. 

As a nice bonus, while we're updating the Compose file, we can also map the runner container's port to a host port. This will enable access to its running FastAPI application's automatically generated `/docs` endpoint to get us a useful quick and human-friendly look at its supported algorithms, in addition to the more complete JSON Schema specs we can get by running the `get_schemas` function from the runner's `models` module manually.

```yaml
runner-discovery:
    container_name: runner-discovery
    image: runner-discovery
    command: python main.py
    environment:
      - PORT=5099

meme-classifier-runner:
    # ...
    environment:
      # ...
      - CONTAINER_NAME=meme-classifier-runner
      - PORT=5000
      - RUNNER_DISCOVERY_CONTAINER_NAME=runner-discovery
      - RUNNER_DISCOVERY_PORT=5099
    ports:
      - 5000:5000

controller:
    # ...
    environment:
      # ...
      - RUNNER_DISCOVERY_CONTAINER_NAME=runner-discovery
      - RUNNER_DISCOVERY_PORT=5099
    # ...
```

### Trying it out

## So, is it really that extensible?

### Adding a new algorithm to an existing runner container

### Adding a new runner container

## Conclusions
 

 
- Desarrollamos `runner` como librería
- El Dockerfile de cada runner va a tener "if prod then instalar de pip else copy e instalar local"

Hicimos
- Armar librería corrible como módulo
- Hay que setearle PYTHONPATH para que encuentre el integration_adapter
- Hay que setearle PORT
- Hay que setearle CONTAINER_NAME
- (ver cuáles env var más por las dudas)
- Falta hacer el componente runner-discovery
- Cuando corre:
    - Primero llama register
        - Register asume que hay handlers que empiezan con run_*
- 
