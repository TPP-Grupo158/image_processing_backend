# Image Processing Backend

Servicio de procesamiento de imágenes, que se encarga de recibir archivos en formato nifti y procesarlos para segmentación de metástasis o acv; o detección de alzheimer.

## Pre-requisitos
- Docker
- 1 o mas GPUS nvidia si se quiere utlizar gpu para las predicciones por lo menos 10 gb de memoria libres
- Para que las predicciones sean acessibles publicamente es necesarios hacer los siguiente:

```
docker exec -it minio mc config host add myminio http:/minio:9000 your-access-key your-secret-key
docker exec -it minio-server mc anonymous set download local/medical-images
```

## Correr con docker

Para construir y levantar el contenedor de docker:

`docker compose -f docker-compose.yml up --build`

Esto levanta el sistema de almacenamiento de objetos (MinIO), y el backend junto con la base de datos (MongoDB).

Para acceder a la UI de MinIO, se debe ingresar la siguiente dirección en un navegador web:
http://localhost:9001/

Para acceder a la base de datos, se puede utilizar un gestor como [MongoDB Compass](https://www.mongodb.com/products/tools/compass)

## Acceso a la documentación (Swagger)

Para acceder al Swagger que contiene la documentación de los endpoints (e incluso para poder probarlos), en un navegador web se debe ingresar a la siguiente dirección:

http://localhost:8000/docs

Si se modifica el número de puerto en el archivo [docker-compose](./docker-compose.yml), entonces se debe modificar el puerto en la dirección mostrada.


## Correr linter dentro docker

`docker compose exec backend_ai ruff check .`