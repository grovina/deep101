# deep101

Execute nesta pasta:
```
docker run -it -v $(pwd):/notebooks -p 8888:8888 tensorflow/tensorflow:1.4.1-py3
```

Ou o análogo com GPU:
```
nvidia-docker run -it -v $(pwd):/notebooks -p 8888:8888 tensorflow/tensorflow:1.4.1-gpu-py3
```

#### TensorBoard

Para executar o TensorBoard fora do container docker:
```
tensorboard --logdir=logs
```

Para rodar o TensorBoard de dentro do container:

- Adicione `-p 6006:6006` ao comando de rodar o docker
- Lance o TensorBoard com o comando acima dentro do container
