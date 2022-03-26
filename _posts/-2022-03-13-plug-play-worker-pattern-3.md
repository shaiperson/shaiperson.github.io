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
