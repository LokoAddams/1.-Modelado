
El programa simula la propagación de calor (o un incendio) sobre un terreno. Para ello, utiliza una famosa ecuación diferencial parcial conocida como **ecuación de reacción-difusión**.

-----
### Supuestos
- La funcion de Temperatura tiene como variables independientes o tambien llamadas de entrada a T(x,y), aunque en el codigo se utiliza las alturas "z" solo es para graficar el terreno. El tiempo es tomado en cuenta en `get_tem()` que aplica el Metodo de Euler para encontrar la solucion numerica.
-----

### La Fórmula Principal

La simulación se rige por la ecuación que se encuentra dentro de la función `get_Tem`:

```python
dTdt = D * laplacian(T) - alpha * T + S
```

Esta línea calcula el **cambio de la temperatura (`dT/dt`)** en un pequeño intervalo de tiempo. Matemáticamente, esto representa la siguiente Ecuación Diferencial Parcial (EDP):

$$\frac{\partial T}{\partial t} = \underbrace{D \nabla^2 T}_{\text{Difusión}} - \underbrace{\alpha T}_{\text{Enfriamiento}} + \underbrace{S(T)}_{\text{Fuente de calor}}$$


Vamos a desglosarla:

1.  **Término de Difusión ($D \nabla^2 T$):**

      * Describe cómo se **propaga o difunde** el calor.
      * `D` es el coeficiente de difusión (qué tan rápido se esparce).
      * $\nabla^2 T$ es el **Laplaciano** de la temperatura. Esta es la parte que calcula la `funcion laplacian`.

2.  **Término de Enfriamiento ($-\alpha T$):**

      * Modela la **pérdida de calor** hacia el ambiente.
      * La temperatura disminuye a una velocidad proporcional ($\alpha$) a la temperatura actual (`T`).

3.  **Término Fuente ($S(T)$):**

      * Representa una **fuente externa de calor**.
      * En tu código, `S = (T > Tign).astype(float) * 30.0` significa que si la temperatura `T` en un punto supera una "temperatura de ignición" `Tign`, se genera una cantidad fija de calor (30.0). Esto es lo que hace que el "fuego" se propague.

-----

###  ¿Qué hacen las funciones `laplacian` y `get_Tem`?

#### **`laplacian(Z)`: Aproximación Numérica del Laplaciano**

Esta función es clave. El operador Laplaciano, $\\nabla^2 T$, en 2D es:
$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}$$
Intuitivamente, **el Laplaciano mide si un punto está más "caliente" o "frío" que el promedio de sus vecinos inmediatos.**

  * Si un punto es mucho más caliente que sus vecinos (un pico), el Laplaciano es negativo y el calor tenderá a difundirse hacia afuera.
  * Si un punto es más frío (un valle), el Laplaciano es positivo y el calor de los vecinos tenderá a fluir hacia él.

Como no se puede calcular la derivada exacta en una grilla de datos, el código usa un **método numérico** llamado **diferencias finitas** para aproximarlo. La línea:

```python
return (np.roll(Z,1,0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z,-1,1)-4*Z)
```

Es la implementación de esta aproximación. Suma los valores de los cuatro vecinos (arriba, abajo, izquierda, derecha) y le resta cuatro veces el valor del punto central. Es una forma muy eficiente de calcular el Laplaciano en toda la grilla a la vez.
### Difusión usa el **Laplaciano**:
  $\nabla^2 T=\frac{\partial^2 T}{\partial x^2}+\frac{\partial^2 T}{\partial y^2}$.
### Metodo por diferencias finitas
$$
  \nabla^2 T\;(i,j)\;\approx\;\frac{T_{i+1,j}-2T_{i,j}+T_{i-1,j}}{\Delta x^2}\;+\;\frac{T_{i,j+1}-2T_{i,j}+T_{i,j-1}}{\Delta y^2}.
$$
- ¿ Por que usamos diferencias finitas y no simplemente usamos la derivada de la parte superior?
Porque no podemos calcular la derivada de una funcion que no conocemos
#### **`get_Tem(T)`: Aplicando el Método de Euler**

Esta función calcula el estado siguiente de la simulación. La línea final es:

```python
return T + dT * dt
```

Esto es exactamente el **Método de Euler** que tienes en tus apuntes.

Tus notas dicen que para una ecuación $y' = f(x)$, el siguiente valor $y\_{n+1}$ se calcula como $y\_{n+1} = y\_n + h \\cdot f(x\_n)$.

En tu código es lo mismo, pero para una EDP:

  * La temperatura en el siguiente instante de tiempo (`T_nuevo`) es...
  * ...la temperatura actual (`T`) más...
  * ...el cambio de temperatura (`dT`, que es $\\frac{\\partial T}{\\partial t}$) multiplicado por un pequeño paso de tiempo (`dt`).

$$T(t+dt) \approx T(t) + \frac{\partial T}{\partial t} \cdot dt$$

-----

### Conexión con Tus Apuntes

  * **Ecuaciones Diferenciales Parciales:** Correcto. Esta es una EDP porque la temperatura `T` depende de más de una variable: el espacio (`x`, `y`) y el tiempo (`t`).
  * **Linealidad:** La ecuación es **no lineal**. La parte de difusión ($D \\nabla^2 T$) y la de enfriamiento ($-\\alpha T$) son lineales. Sin embargo, el término fuente `S` depende de una condición (`T > Tign`), lo que lo hace no lineal. Por eso, resolverla analíticamente es muy difícil y se usan métodos numéricos.
  * **Métodos Numéricos:** Tu programa es un ejemplo perfecto de cómo se usan los métodos numéricos para resolver ecuaciones complejas. En lugar de encontrar una fórmula exacta para `T(x,y,t)`, simulas su evolución paso a paso usando el **Método de Euler**.

----

## ¿Cuales son las limitaciones de nuestro modelo?
Estamos simulando un terreno con las siguientes supuestos:
  - Todas las zonas estan cubiertas por el mismo combustible o al menos varios tipos de combustibles con la misma temperatura de ignicion.
    - ¿Que es la temperatura de ignicion?
    Es una temperatura al la cual algun combustible puede mantenerse constante, si necesidad de ser alimentado por otra fuente de calor cercana.
    - ¿Entonces acaso en un bosque e todos los lugares hay algun combustible?
    No realmente, ya que hay zonas donde hay rocas sin ningun combustible encima, pero cabe aclarar que cuando nos referimos a combustible puede ser cualquier otra fuente organica o inorganica que se encuentre en nuestro ambiente. Por ejemplo la tierra no es consumida por el fuego pero si los elementos sobre ella como algun pasto seco, maderas,etc. Todos estos combustibles llegan un punto en los que son consumidos totalmente.
    - ¿Solo los combustibles tienen temperatura de ignición?
    Sí, exactamente. El concepto de "temperatura de ignición" solo se aplica a las sustancias que pueden sufrir combustión, es decir, a los combustibles.
  - Aunque las visualizaciones estan en 3d al momento de hacer la simulacion no tomamos en cuenta como afecta las diferentes alturas del terreno a nuestra simulacion.