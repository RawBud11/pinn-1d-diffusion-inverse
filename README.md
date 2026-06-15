# PINNs — Problema Inverso de Parámetros
### Proyecto Final · Análisis Numérico de Ecuaciones Diferenciales

> **Pista B:** Recuperación del coeficiente de difusividad λ en la ecuación de difusión 1D a partir de observaciones dispersas y ruidosas, usando Physics-Informed Neural Networks (PINNs).

---

## Descripción

Este proyecto implementa una PINN para resolver el **problema inverso de parámetros** en la ecuación de difusión unidimensional:

$$u_t = \lambda \, u_{xx}, \qquad (x,t) \in [0,1]^2$$

con condición inicial $u(x,0) = \sin(\pi x)$ y condiciones de contorno Dirichlet homogéneas. El coeficiente $\lambda > 0$ es **desconocido** y se recupera conjuntamente con el campo $u(x,t)$ a partir de $N_d = 100$ observaciones con 1% de ruido gaussiano.

**Resultado principal:** error relativo en $\hat{\lambda}$ de **0.17%** tras 8000 épocas de entrenamiento.

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `pinn_1d_diffusion_inverse.ipynb` | Notebook principal — problema directo, inverso, validación y estudio de sensibilidad |
| `Proyecto_Final_PINNs_PistaB.pdf` | Informe completo del proyecto |
| `Proyecto_Final_PINNs_PistaB.tex` | Código fuente LaTeX del informe |
| `LICENSE` | Licencia del repositorio |

---

## Resultados

| Métrica | Valor |
|---|---|
| λ real | 0.10000 |
| λ estimado | 0.10017 |
| Error relativo en λ | **0.17%** |
| Error L² del campo (prob. inverso) | 1.57×10⁻³ |
| Error L² PINN vs analítica (prob. directo) | 1.47×10⁻³ |

---

## Método

- **Red neuronal:** 3 capas ocultas × 40 neuronas, activación `tanh`, inicialización Glorot
- **λ como variable entrenable:** `tf.Variable` optimizado conjuntamente con los pesos
- **Función de pérdida:** residuo PDE + condición inicial + contorno + ajuste a datos
- **Derivadas:** diferenciación automática con tapes anidados (`tf.GradientTape`)
- **Optimizador:** Adam, η = 10⁻³, 8000 épocas
- **Positividad:** proyección explícita λ ← max(λ, 10⁻⁶) tras cada paso

---

## Validación

1. **Comparación con Crank-Nicolson** — el PINN supera en precisión al esquema clásico (error L² 1.47×10⁻³ vs 6.71×10⁻³)
2. **Estudio de sensibilidad** — $N_d \in \{20, 50, 100, 200\}$, ruido $\in \{0\%, 1\%\}$, 3 semillas independientes; error en λ < 1% en los 24 experimentos
3. **Plausibilidad física** — positividad de λ garantizada; señal de gradiente consistente durante todo el entrenamiento

---

## Requisitos

```bash
python >= 3.8
tensorflow >= 2.10
numpy
matplotlib
scipy
```

Instalación:
```bash
pip install tensorflow numpy matplotlib scipy
```

---

## Ejecución

El notebook está diseñado para ejecutarse de arriba a abajo en una sola sesión:

```bash
jupyter notebook pinn_1d_diffusion_inverse.ipynb
```

O directamente en **Google Colab** — todas las dependencias están disponibles por defecto.

> **Semillas fijas:** `np.random.seed(7)` y `tf.random.set_seed(42)` garantizan reproducibilidad exacta.

---

## Referencia principal

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
*Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*
Journal of Computational Physics, 378, 686–707.

---

## Licencia

Este proyecto está bajo la licencia especificada en el archivo `LICENSE`.
