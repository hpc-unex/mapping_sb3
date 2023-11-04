# mapping_sb3

Repositorio creado para el desarrollo de un agente de aprendizaje por refuerzo cuya tarea será la de obtener la mejor distribución de procesos en un número de nodos con capacidad variable, optimizando el número de comunicaciones entre ellos.

<h2> Ejecución </h2>

El código se ejecuta con la siguiente línea de comandos:

```
python main.py
```

De forma predeterminada carga el fichero de simulación "binomial_4_2.json", aunque esto se modificará para pasar el fichero deseado por parámetros.

La descripción de los parámetros es la siguiente:

- Graph: Esta sección describe las propiedades del grafo utilizado en el contexto de la simulación o experimento.

- P: Número 4. Representa alguna característica del grafo.
- M: Número 2. Otra característica del grafo.
- m: Número 1048576 (2^20). Posiblemente un tamaño o límite relacionado con el grafo.
- S: Número 16384 (2^14). Posiblemente otro tamaño o límite relacionado con el grafo.
- root: Número 0. El nodo raíz del grafo.
- node_names: Una lista ["M0", "M1"] que asigna nombres a los nodos.
- capacity: Una lista [2, 2] que parece representar las capacidades de los nodos.
- comms: Una subsección que describe las propiedades de las comunicaciones en el grafo.
    - edges: Lista de listas que representa las conexiones entre nodos.
    - volume: Lista de volúmenes de comunicación asociados a cada conexión.
    - n_msgs: Lista de números de mensajes asociados a cada conexión.
    - opt_nodes_feats: Diccionario que describe características opcionales de los nodos.
    - opt_edges_feats: Diccionario que describe características opcionales de las conexiones.



<h2> Entorno </h2>

Se ha desarrollado un entorno que simule la estructura de los nodos y los procesos mencionados anteriormente. Como es un entorno basado en la libreria "Gym", sus parámetros por defecto son los siguientes:

- Action Space: Es la acción que puede tomar el agente. Ejemplo: Pongo el proceso 1 en el nodo 0. Se trata de un tipo de dato Discreto que puede tomar valores entre 0 y n_procesos*n_nodos.

- Observation Space: Es la ventana de observación que tiene el agente para decidir qué decisión toma en el siguiente instante. En este caso, es un vector de tantas posiciones como procesos hay que distribuir donde dichas posiciones están inicializadas a n_nodos+2. Representa la capacidad restante que tienen los nodos. Se suma +2 porque no puede haber un "0" en el espacio de observación que se da como entrada a la red neuronal, ya que da error.

- Current assignment: Es el estado actual del clúster, es un vector de tantas casillas como procesos, inicializadas a n_nodos+1.

<h2> Funcionamiento </h2>

La idea actual es que el agente cada vez que coloque dos o mas procesos, el programa vaya al grafo de comunicaciones, compruebe si los procesos que hay desplegados se comunican entre sí, y de ser así, se realice la comprobación de:

- Si están en el mismo nodo: el coste de comunicación es 0.
- Si están en nodos distintos: el coste de comunicación es el que diga la matriz de adyacencia.

Esta sería la recompensa que recibiría el agente, la cual hay que minimizar, como consecuencia se minimizarían las comunicaciones.

<h2> Stable Baselines 3 </h2>

Esta es una librería que permite aplicar algoritmos famosos como política de aprendizaje al aprendizaje por refuerzo. Actualmente se ha utilizado Proximal Policy Optimization y DQN.

https://stable-baselines3.readthedocs.io/en/master/
