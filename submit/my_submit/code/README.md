
## Final Test Code

- build
```shell
docker build -t sn8/baseline:sub -f ./Dockerfile  .
```

- run
```shell
docker run -it --shm-size=64g --gpus all -v $PWD:/workdir/ --rm sn8/baseline:sub bash
```

- inference
```shell
chmod +x test.sh
./test.sh {data_folder} {output_file}
```

- exapmle `open public test`
```shell
./test.sh data/Louisiana-West_Test_Public ./
```

as sample 1 file, `code/data/Louisiana-West_Test_Public`