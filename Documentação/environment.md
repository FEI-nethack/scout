
O código de criação environment.yml, pode ser entendido como visto abaixo:

```yaml
#escolhe o nome:
name: nle   
#dependencias a serem instaladas (instala o python e o pip): 
dependencies:
  - python=3.8
  - pip
  #dependencias a serem instaladas pelo pip (nle e minihack):  
  - pip:
    - nle
    - minihack
```
