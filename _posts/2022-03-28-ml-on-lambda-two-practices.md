---
layout: post
title: "ML on AWS Lambda: Two Practices"
subtitle: "Avoid the filesystem trap"
permalink: ml-on-aws-lambda-two-practices
highlights:
  - AWS Lambda
  - AWS S3
---

* TOC
{:toc}

## Introduction

I'd like to share a pattern that emerged in my work deploying machine learning (ML) inference Python code to AWS Lambda. To do this, I'll first offer three related observations. Following, we'll derive two practices from those observations that you might want to consider for your own processes.

## Observations

### 1. On Lambda and its filesystem feature
AWS Lambda can be a convenient way of deploying code to production for many use cases. Of course, it also comes with limitations. In particular, its serverless and flexible nature entails a local storage constraint: each Lambda execution environment gets a filesystem on an ephemeral storage space for it to use that is available under `/tmp` and has a hard limit of 512 MB in size (see [docs](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)).

### 2. On using popular ML libraries  

Popular ML libraries that offer pre-trained models (such as [Hugging Face](https://huggingface.co/models), [OpenAI's CLIP](https://github.com/openai/CLIP) or [JaidedAI's EasyOCR](https://github.com/JaidedAI/EasyOCR)) commonly use your filesystem to download models to and to use as cache. Further, these libraries may differ in the default filesystem paths they'd use as cache, but they usually expose a way to configure the paths they'll use. On the other hand, as far as I can tell at this point in time, they don't usually offer a way to stream models into RAM. Using your filesystem as a means for caching models is convenient for development processes as it allows for a smoother iteration on code, but it also means we're forced to rely on a filesystem, and one with sufficient available space at that, to be able to get our models downloaded.

Also quite common is for single models or the conjunction of several models to exceed 512 MB.

### 3. On downloading objects from S3

When you need to get an object from S3, you often just need to download it to disk. This seems intuitive and easy, and is a common practice. You may even do this when you only need the object in RAM, by downloading it first to your filesystem and then loading it into your program from there:

```python
import boto3
s3_resource = boto3.resource('s3')

s3_object = s3_resource.Object('my-bucket', 'my_file.txt')
with open('temp_my_file.txt', 'w') as f:
    s3_object.download_fileobj(f)

# do something with temp_my_file.txt, remember to delete it if appropriate.
```

However, if you're going to download the S3 object to disk only to then load and use it in RAM, then you'd probably prefer a way to get the object into RAM directly.

## Deriving practices

Deploying ML inference code is a worthy use case for Lambda. But, in light of popular libraries relying on disk space for getting you pre-trained models, if you use enough of these then your lambdas are liable to eventually run into the 512 MB limit. This would be the case even if you don't download pre-trained models but you do rely on getting models (either pre-downloaded, trained by you or whatever the case may be) from a remote storage service with which you're interacting in a similar way to what I showed before with S3.

So, to solve this potential and likely issue, two good practices you may want to adopt follow.

### 1. Make libraries' on-disk cache path configurable by environment

Either on a model-by-model basis or for all of your library-downloaded models, leverage these libraries' options to configure cache paths and use environment variables to set them. If you work with a separate ML team that writes your Lambda-bound code, encourage them to adopt this practice on your behalf. That way, when you deploy your code to Lambda, you'll be able to easily have those paths stem from `/tmp` and to avoid "read-only filesystem" errors coming from your libraries attempting to write to off-limits paths. This won't save you from running into the storage space limit, but it will make deployment easier while your space usage is within bounds.

To view this in an example, let's assume our code uses CLIP's `ViT-B/32` and Hugging Face's `bert-base-cased`. This means our code at a certain point might include some lines like:

```python
import clip
from transformers import BertModel

model, preprocess = clip.load('ViT-B/32')
bert = BertModel.from_pretrained('bert-base-cased')
```

The default cache dirs used by `clip` and `transformers` in these function calls are (at this point in time) under `~/.cache`, namely `~/.cache/torch/transformers` and `~/.cache/clip` respectively. To adopt this practice, you'd set something like a `MODEL_CACHE_FS_PATH` environment variable to a path starting with `/tmp` and use those libraries' cache path configuration options:

```python
import os
import clip
from transformers import BertModel

configured_cache_path = os.environ.get('MODEL_CACHE_FS_PATH', './cache')

model, preprocess = clip.load('ViT-B/32', download_root=configured_cache_path)
bert = BertModel.from_pretrained('bert-base-cased', cache_dir=configured_cache_path)
```

### Use S3 to store models and stream them into RAM

Once your library-downloaded models (or other large files you may need) conspire to exceed the 512 MB to your lambdas, you'll need a way to download them that does not require a filesystem. S3 is indeed a good option, since it does offer an easy way to stream objects directly to RAM. If you're deploying Python code to Lambda, this is very easy to do using `boto3`. The way you might take advantage of this option is to pre-download the models you've been getting through your libraries' API, serialize them (e.g by using `pickle` or library-specific serialization APIs), upload the serialized files to S3 and then fetch those objects in your Lambda-bound code in a way that gets them into RAM directly. Note, of course, that this is useful for models you get from other sources as well, such as the ones you train yourself. 

To quickly look at an example for this, let's assume we're using PyTorch and you've pickled and uploaded the `state_dict` of an instance of `Model` to S3 at `models-bucket/model_state_dict.pkl`. This is what your code might look like.

```python
import io
import pickle

import boto3
s3_resource = boto3.resource('s3')

# Stream pickled `state_dict` into variable rather than save to disk
bytes_stream = io.BytesIO()
s3_object = s3_resource.Object('models-bucket', 'model_state_dict.pkl')
s3_object.download_fileobj(bytes_stream)
pickled_state_dict = bytes_stream.getvalue()

# Load model
state_dict = pickle.loads(pickled_state_dict)
model = Model()
model.load_state_dict(state_dict)
model.eval()

# Have fun with model
```

### A final note

Streaming objects from S3 into RAM is nothing new nor is it too much of an obscure functionality if you look at the docs. However, it seems worth highlighting in the particular context of deploying ML code to Lambda. This is an increasingly popular go-to for ML deployment, and these practices easily solve an issue that anyone starting to adopt Lambda for ML is bound to encounter.


