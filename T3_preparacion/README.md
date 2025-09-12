# T3 – Preparación / Ingeniería de Datos

Ver instrucciones dentro.
# T3 – Preparación / Ingeniería de Datos

**Propósito.** Dejar un dataset **limpio y modelable**, documentando cada decisión.

## Inputs
- `../data/incident_event_log.csv` (crudo; NO versionado)
- `../diccionario.txt` (si aplica)

## Output
- `incidents_clean_for_model.csv` (dataset listo para modelado)

## Cómo reproducir
1) Colocar el crudo en `../data/incident_event_log.csv`.  
2) Abrir `T2yT3.ipynb` y ejecutar todas las celdas.  
3) Se genera `incidents_clean_for_model.csv`.

## Pipeline (resumen de decisiones)
- **Tipos y fechas:** parseo de `opened_at` (y otras) a `datetime`.
- **Duplicados:** criterio (p. ej. por `number`) y tratamiento.
- **Faltantes:** qué variables se imputan / descartan y por qué.
- **Aberrantes/Atípicos:** **NO se descartan**; se marca `ttr_outlier` y/o se crea `ttr_h_winsor` para modelado posterior (política del curso).
- **Features derivadas:** `ttr_h`, `ttr_h_log`, etc. (si aplica).
- **Normalización/encoding:** **no aquí**; se hará en T4 dentro de un `Pipeline`.

## Diccionario (salida limpia) – completar
- `ttr_h`: tiempo de resolución (h)
- `ttr_outlier`: {0/1} indicador de outlier de `ttr_h`
- `priority_level`, `impact_level`, `urgency_level`: …
- `category`, `subcategory`, `contact_type`: …
- `opened_at`: fecha apertura
- `number`: id del ticket
